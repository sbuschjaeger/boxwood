// The Python extension module boxwood._boxwood. It only moves data across the
// language boundary: arrays in, arrays out, errors as boxwood.Error (a subclass
// of ValueError). All numeric work happens in the C++ library.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include "boxwood/ensemble.hpp"

namespace nb = nanobind;
using namespace boxwood;

namespace {

/// The input matrix: n observations of n_numeric float32 features, C-contiguous.
using Input = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

/// Puts `v` into a 2-D numpy array that owns the buffer: a capsule deletes the
/// heap copy of the vector once the array and all its views are gone.
template <class T>
nb::ndarray<T, nb::numpy, nb::ndim<2>, nb::device::cpu> owned(std::vector<T>&& v, size_t n,
                                                              size_t cols) {
  auto* buf = new std::vector<T>(std::move(v));
  if (buf->empty()) buf->resize(1);  // keep the DLPack data pointer non-null
  nb::capsule owner(buf, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  return nb::ndarray<T, nb::numpy, nb::ndim<2>, nb::device::cpu>(buf->data(), {n, cols}, owner);
}

const char* task_name(Task t) {
  return t == Task::Regression ? "regression" : t == Task::Binary ? "binary" : "multiclass";
}

const char* source_name(uint8_t source) {
  return source == 0 ? "numeric" : source == 1 ? "onehot" : "ctr";
}

const char* leaf_type_name(LeafType t) { return t == LeafType::Int8 ? "int8" : "float32"; }

/// The FitReport as the dict the Python layer returns.
nb::dict report_dict(const FitReport& r) {
  nb::dict d;
  d["objective_start"] = r.objective_start;
  d["objective"] = r.objective;
  d["holdout_loss"] = r.holdout_loss;
  d["iterations"] = r.iterations;
  d["converged"] = r.converged;
  d["lambda"] = r.lambda;
  d["no_gain"] = r.no_gain;
  d["holdout_loss_start"] = r.holdout_loss_start;
  d["splits_used"] = r.splits_used;
  d["trees_removed"] = r.trees_removed;
  nb::list path;
  for (const PathStep& s : r.path) {
    nb::dict e;
    e["lambda"] = s.lambda;
    e["objective"] = s.objective;
    e["holdout_loss"] = s.holdout_loss;
    e["iterations"] = s.iterations;
    path.append(e);
  }
  d["path"] = path;
  return d;
}

/// The categorical columns as Python passes them, dictionary-encoded: per
/// categorical feature a pair (n uint32 codes, the list of distinct strings);
/// empty for a numeric model.
using Codes = nb::ndarray<const uint32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using CatColumns = std::vector<std::pair<Codes, std::vector<std::string>>>;

/// string_views over a CatColumns, plus the spans a Batch needs. The views own the
/// references and must outlive the Batch that reads the spans.
struct CatSpans {
  std::vector<std::vector<std::string_view>> views;
  std::vector<std::span<const std::string_view>> spans;
  std::vector<std::span<const uint32_t>> codes;
};

CatSpans make_cat_spans(const CatColumns& cat) {
  CatSpans cs;
  cs.views.resize(cat.size());
  for (size_t f = 0; f < cat.size(); ++f) {
    const auto& [codes, values] = cat[f];
    for (const std::string& s : values) cs.views[f].emplace_back(s);
    cs.spans.emplace_back(cs.views[f]);
    cs.codes.emplace_back(codes.data(), codes.size());
  }
  return cs;
}

/// The Batch of numeric observations `x` and the categorical columns `cat`.
Batch make_batch(const Input& x, const CatSpans& cat) {
  const size_t n = x.shape(0);
  Batch b{std::span<const float>(x.data(), n * x.shape(1)), n, cat.spans};
  b.categorical_codes = cat.codes;
  return b;
}

/// A 1-D float64 array (targets, weights).
using Vec = nb::ndarray<const double, nb::ndim<1>, nb::c_contig>;

/// A Batch over arrays that the caller keeps alive.
Batch make_batch(const Input& x, const CatSpans& cat, const Vec& y, const Vec& w) {
  Batch b = make_batch(x, cat);
  b.y = std::span<const double>(y.data(), y.size());
  b.weight = std::span<const double>(w.data(), w.size());
  return b;
}

/// The optimizer named `name` (the names of optimizer_name).
OptimizerKind optimizer_kind(std::string_view name) {
  for (const OptimizerKind k : {OptimizerKind::TreeNewton, OptimizerKind::Adam, OptimizerKind::GradientDescent}) {
    if (name == optimizer_name(k)) return k;
  }
  throw nb::value_error("unknown optimizer; use tree_newton, adam or gradient_descent");
}

/// The strategy named `name`: "fixed", "path" or "early_stop".
Strategy strategy_kind(std::string_view name) {
  if (name == "fixed") return Strategy::Fixed;
  if (name == "path") return Strategy::Path;
  if (name == "early_stop") return Strategy::EarlyStop;
  throw nb::value_error("unknown strategy; use fixed, path or early_stop");
}

/// The options of a fit; the strategy decides whether a run stops on the holdout.
FitOptions make_options(double lam, std::string_view optimizer, double learning_rate, int max_iter,
                       double tol, int patience, std::string_view strategy,
                       std::optional<uint32_t> split_budget, std::optional<uint64_t> max_bytes) {
  FitOptions opt{lam, optimizer_kind(optimizer), learning_rate, StopRule{max_iter, tol, false, patience},
                strategy_kind(strategy)};
  opt.split_budget = split_budget;
  opt.max_bytes = max_bytes;
  return opt;
}

/// Runs `method` of `self` on the observations `x` with the GIL released, into a fresh
/// n x `cols` buffer, and returns it as a numpy array that owns the buffer.
/// The categorical strings are turned into string_views before the release.
template <class T>
nb::ndarray<T, nb::numpy, nb::ndim<2>, nb::device::cpu> compute(
    const ObliviousTreeEnsemble& self, void (ObliviousTreeEnsemble::*method)(const Batch&, std::span<T>) const,
    const Input& x, const CatColumns& cat, size_t cols) {
  const size_t n = x.shape(0);
  std::vector<T> out(n * cols);
  const CatSpans cs = make_cat_spans(cat);
  const Batch batch = make_batch(x, cs);
  {
    nb::gil_scoped_release release;
    (self.*method)(batch, std::span<T>(out));
  }
  return owned(std::move(out), n, cols);
}

}  // namespace

NB_MODULE(_boxwood, m) {
  m.doc() = "boxwood's native core; use the boxwood package instead";

  nb::exception<boxwood::Error>(m, "Error", PyExc_ValueError);

  m.def("backends", &boxwood::available_backends,
        "Names of the prediction backends available on this CPU, fastest first.");

  nb::class_<ObliviousTreeEnsemble>(m, "Ensemble")
      .def_static("from_catboost_json", &ObliviousTreeEnsemble::from_catboost_json,
                  nb::arg("json"))
      .def_static(
          "from_bytes",
          [](nb::bytes data) {
            return ObliviousTreeEnsemble::from_bytes(std::span<const std::byte>(
                static_cast<const std::byte*>(data.data()), data.size()));
          },
          nb::arg("data"))
      .def(
          "to_bytes",
          [](const ObliviousTreeEnsemble& self) {
            const std::vector<std::byte> v = self.to_bytes();
            return nb::bytes(v.data(), v.size());
          })
      .def(
          "predict_margin",
          [](const ObliviousTreeEnsemble& self, Input x, CatColumns categorical) {
            return compute(self, &ObliviousTreeEnsemble::predict_margin, x, categorical,
                           self.info().n_outputs);
          },
          nb::arg("X"), nb::arg("categorical"))
      .def(
          "predict",
          [](const ObliviousTreeEnsemble& self, Input x, CatColumns categorical) {
            return compute(self, &ObliviousTreeEnsemble::predict, x, categorical, self.info().n_outputs);
          },
          nb::arg("X"), nb::arg("categorical"))
      .def(
          "leaf_index",
          [](const ObliviousTreeEnsemble& self, Input x, CatColumns categorical) {
            return compute(self, &ObliviousTreeEnsemble::leaf_index, x, categorical, self.info().n_trees);
          },
          nb::arg("X"), nb::arg("categorical"))
      .def(
          "fit",
          [](ObliviousTreeEnsemble& self, Input x, CatColumns categorical, Vec y, Vec weight,
             Input x_hold, CatColumns categorical_hold, Vec y_hold, Vec weight_hold, double lam,
             std::string_view optimizer, double learning_rate, int max_iter, double tol, int patience,
             std::string_view strategy,
             std::optional<uint32_t> split_budget, std::optional<uint64_t> max_bytes) {
            const CatSpans cat = make_cat_spans(categorical);
            const CatSpans hcat = make_cat_spans(categorical_hold);
            const Batch train = make_batch(x, cat, y, weight);
            const Batch hold = make_batch(x_hold, hcat, y_hold, weight_hold);
            const FitOptions opt =
                make_options(lam, optimizer, learning_rate, max_iter, tol, patience, strategy, split_budget,
                             max_bytes);
            FitReport r;
            {
              nb::gil_scoped_release release;
              r = self.fit(train, hold, opt);
            }
            return report_dict(r);
          },
          nb::arg("X"), nb::arg("categorical"), nb::arg("y"), nb::arg("weight"), nb::arg("X_hold"),
          nb::arg("categorical_hold"), nb::arg("y_hold"), nb::arg("weight_hold"), nb::arg("lam"),
          nb::arg("optimizer"), nb::arg("learning_rate"), nb::arg("max_iter"), nb::arg("tol"),
          nb::arg("patience"), nb::arg("strategy"), nb::arg("split_budget"), nb::arg("max_bytes"))
      .def(
          "update",
          [](ObliviousTreeEnsemble& self, Input x, CatColumns categorical, Vec y, Vec weight,
             int steps) {
            const CatSpans cat = make_cat_spans(categorical);
            const Batch window = make_batch(x, cat, y, weight);
            UpdateReport r;
            {
              nb::gil_scoped_release release;
              r = self.update(window, steps);
            }
            nb::dict d;
            d["objective_start"] = r.objective_start;
            d["objective"] = r.objective;
            d["steps"] = r.steps;
            return d;
          },
          nb::arg("X"), nb::arg("categorical"), nb::arg("y"), nb::arg("weight"), nb::arg("steps"))
      .def("set_backend", &ObliviousTreeEnsemble::set_backend, nb::arg("name"))
      .def("set_threads", &ObliviousTreeEnsemble::set_threads, nb::arg("n"))
      .def(
          "set_leaf_type",
          [](ObliviousTreeEnsemble& self, std::string_view name) {
            if (name == "int8") {
              self.set_leaf_type(LeafType::Int8);
            } else if (name == "float32") {
              self.set_leaf_type(LeafType::Float32);
            } else {
              throw nb::value_error("leaf type must be \"int8\" or \"float32\"");
            }
          },
          nb::arg("name"))
      .def(
          "info",
          [](const ObliviousTreeEnsemble& self) {
            const Info i = self.info();
            nb::dict d;
            d["task"] = nb::str(task_name(i.task));
            d["n_outputs"] = i.n_outputs;
            d["n_numeric"] = i.n_numeric;
            d["n_categorical"] = i.n_categorical;
            d["n_trees"] = i.n_trees;
            d["n_splits"] = i.n_splits;
            d["tree_bytes"] = i.tree_bytes;
            d["leaf_type"] = nb::str(leaf_type_name(i.leaf_type));
            d["numeric_flat_index"] = nb::cast(i.numeric_flat_index);
            d["categorical_flat_index"] = nb::cast(i.categorical_flat_index);
            d["fit_lambda"] = i.fit_lambda;
            d["fit_sigma"] = i.fit_sigma;
            d["fit_optimizer"] = i.fit_optimizer;
            return d;
          })
      .def(
          "splits",
          [](const ObliviousTreeEnsemble& self, uint32_t tree) {
            nb::list out;
            for (const ObliviousTreeEnsemble::Split& s : self.splits(tree)) {
              out.append(
                  nb::make_tuple(source_name(s.source), s.index, s.kind == 1 ? "eq" : "gt", s.arg));
            }
            return out;
          },
          nb::arg("tree"))
      .def("leaves", &ObliviousTreeEnsemble::leaves, nb::arg("tree"))
      .def("bias", &ObliviousTreeEnsemble::bias);
}

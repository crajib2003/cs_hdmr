# Model Computation

* sym_poly_utils.sample_honb(sample, base_syms, ord_cap=3, deg_cap=3)

  Constructs a hierarchically orthonormal polynomial basis (honb) given a
  NumPy array of sample points, a list of SymPy symbols to use as variables
  in the basis polynomials, and the bounds on the order (i.e. number of
  variables) and the degree of the polynomials (ord_cap and deg_cap,
  respectively).

  The output is a polynomial basis, in the form of a dictionary whose keys
  are sets of variables, and whose values are lists of polynomials containing
  exactly the set of variables that is the key. In other words, if the
  polynomial variables are x1, x2, x3, ... then the basis might look like:

  {
      (): [1],
      (x1): [x1, x1**2, ...],
      (x2): [x2, x2**2, ...],
      (x1, x2): [x1*x2, x2*x1**2, x1*x2**2, ...],
      ...
  }

  Note that the polynomials in the values are all SymPy Polynomial objects,
  which in particular means that they can be evaluated as functions and also
  manipulated arithmetically as polynomials without any type conversion.

* cs_hdmr.cs_hdmr_expansion(inputs, outputs, basis_dict)

  Constructs a CS-HDMR expansion given a list of input vectors, a list of
  output values, and a basis dictionary (whose structure is that described
  above). The expansion is return as a generic_models.BasisExpansion
  object, which is both callable (i.e. it can be called as if it was a
  function with a list or NumPy array of an input vector and will return
  the output it needs), and able of independently calculating sensitivity
  indices, component polynomial functions, and various other utility
  computations on an HDMR model.

* cs_hdmr.cs_hdmr_modeler(inputs, outputs, ord_cap=3, deg_cap=3)

  A wrapper around cs_hdmr_expansion which uses the sample_honb function
  described above to construct a hierarchically orthonormal basis (taking
  the given ord_cap and deg_cap values), and then runs cs_hdmr_expansion
  on the given inputs and outputs using that basis. This is particularly
  convenient because it only requires the two relevant inputs and no other
  details or technical miscellanea---it can be called and will work correctly
  with just the NumPy arrays of the inputs and outputs.

# Model Validation

  For validation, the above "modeler function" is the key object: it fully
  encapsulates the transition from an input/output pair of data arrays to
  an HDMR model. The model_validation file provides methods for constructing
  samples from functions to use for testing, and also facilities for
  cross-validating samples and calculating the relative L2 errors of the
  approximations of the models on each subdivision of the sample.

  * model_validation.modeler_l2_errors(modeler, sample, sample_divisions=2)

  This method divides the sample into sample_divisions subsets for cross-
  validation, then applies the modeler to each subset and calculates a model,
  and returns the relative L2 errors of the model on each pair of subsets
  (i.e. each possible way to train a model on one subset and test it on
  another). 

  The test files in the directory /tests also automate some of this;
  individual tests can be run with a command like:

  > python -m unittest tests.test_cs_hdmr.TestCSHDMR.test_corr_poly_1

  The component functions are logged for easier inspection in case that they
  have "true values" to compare against.
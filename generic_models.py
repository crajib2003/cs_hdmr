import numpy as np
import sym_poly_utils
import model_validation
import operator

class BasisExpansion:
    def __init__(self, basis_coeff_dict, sample):
        self.basis_coeff_dict = basis_coeff_dict
        self.basis_coeff_list = sum(basis_coeff_dict.values(), [])
        self.sym_sets = self.basis_coeff_dict.keys()
        self.sample = sample
        self.full_model_poly = np.sum(
            [p*c for p, c in self.basis_coeff_list]
        )
        self.component_polys_dict = {
            syms: np.sum([p*c for p, c in coeffs_list])
            for syms, coeffs_list in self.basis_coeff_dict.items()
        }

    def __call__(self, x):
        return self.full_model_poly(tuple(x))
    
    def cosparsity_index(self, sample, threshold=1.0e-5):
        """
        Returns the number of basis polynomials whose dot product with the
        model polynomial over the sample given is above the threshold.
        """
        basis_polys = [p for p, _ in self.basis_coeff_dict]
        dot_prods = [
            sym_poly_utils.sample_dot(
                sample, self.full_model_poly, p, normalize=True
            )
            for p in basis_polys
        ]
        return len([d for d in dot_prods if d > threshold])
    
    def gramian(self):
        """
        Returns a dictionary of the sensitivities of the expansion to each
        cooperative set of variables.
        """
        sym_sets = self.component_polys_dict.keys()
        n = len(sym_sets)

        def get_entry(i, j):
            s1 = sym_sets[i]
            s2 = sym_sets[j]
            
            p1 = self.component_polys_dict[s1]
            p2 = self.component_polys_dict[s2]

            return sym_poly_utils.sample_dot(
                self.sample, p1, p2, normalize=True
            )

        gramian = np.array(
            [
                [ get_entry(i, j) for j in range(n) ]
                for i in range(n)
            ]
        )

        return gramian, sym_sets

    def sensitivity_dict(self, sample_out):
        sample_norm = np.sum(sample_out**2) / sample_out.shape[0]
        G, sym_sets = self.gramian()
        
        null_ix = sym_sets.index(())
        null_var = sum(G[null_ix, :])
        norm_val = sample_norm - null_var
        
        sym_sets.pop(null_ix)

        n = len(self.sample)
        
        d = {}
        for i in range(len(sym_sets)):
            structural = G[i, i] / norm_val * n
            correlative = (sum(G[i, :i]) + sum(G[i, (i+1):])) / norm_val * n
            d[sym_sets[i]] = {
                's': structural,
                'c': correlative,
                't': structural + correlative
            }

        return d

    def component_functions(self, coeff_gate=0.0):
        """
        Returns a dictionary of the HDMR component (polynomial) functions
        as symbolic polynomials for each set of variables.
        """
        return {
            syms: sum(
                [p*c for p, c in polys if np.absolute(c) >= coeff_gate]
            )
            for syms, polys in self.basis_coeff_dict.items()
        }

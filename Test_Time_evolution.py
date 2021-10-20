import Time_evolution as te

deltat = 0.1
nt_steps = 100
np_steps = 100
pmin = 0.01
pmax = 0.2
qs = 0.1

ev = te.Evolution(deltat, nt_steps, pmin, pmax, np_steps, Qs=qs)

def test_initial_condition():

    fun, num = ev.initial_condition()

    assert len(fun) == len(num)

    return


def test_derivative():

    deriv, Ia, Ib, T_start = ev.derivative()

    Ia_test = 0
    for i in range(ev.lattice.get_p_lattice()):
        Ia_test += ev.deltap_[i] * ev.lattice.get_p_lattice()[i]**2 * ev.function[i] * (1 + ev.function[i])

    assert isinstance(Ia, float)
    assert np.round(Ia, 7) == np.round(Ia_test, 7)

    Ib_test = 0
    for i in range(ev.lattice.get_p_lattice()):
        Ib_test += 2 * ev.deltap_[i] * ev.lattice.get_p_lattice()[i] * ev.function[i]

    assert isinstance(Ib, float)
    assert np.round(Ib, 7) == np.round(Ib_test, 7)

    T_start_test = Ia_test / Ib_test

    for i in range(len(ev.function) - 1):
        mom_deriv = (ev.function[i+1] - ev.function[i]) / ev.deltap[i]
        deriv_test = ev.qhat * ev.lattice.get_p_lattice()[i]**2 * (mom_deriv + ev.function[i] * (1 + ev.function[i]) / T_start_test)

        assert isinstance(deriv_test, float)
        assert  np.round(deriv_test, 7) == np.round(deriv[i], 7)

    return

test_initial_condition()
test_initial_condition()

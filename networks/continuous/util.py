import numpy as np

def accuracy_per_stimulus_window(Y_pred: np.ndarray, L_stimuli: np.ndarray, onset_times, D):
    """
    For each stimulus k, average predicted outputs over its window [onset, onset + D[k]),
    then take argmax and compare to stimulus label.
    """
    correct = 0
    for k, t0 in enumerate(onset_times):
        t1 = t0 + int(D[k])
        y_avg = Y_pred[t0:t1].mean(axis=0)
        pred_cls = y_avg.argmax()
        true_cls = L_stimuli[k].argmax()
        correct += int(pred_cls == true_cls)
    return correct / len(onset_times)

def build_stream_sinusoidal_pulsed_window_labeled(
    S: np.ndarray,          # (K, n_inputs) stimulus vectors
    D: np.ndarray,          # (K,) delay lengths
    SL: np.ndarray,         # (K,) stimulus lengths (<= D)
    L: np.ndarray,          # (K, n_outputs) one-hot labels
    dt: float = 0.1,
    phase_reset: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a continuous stream where each stimulus k is applied only for SL[k] timesteps,
    then input is set to zero until the end of the delay window D[k].
    Labels are repeated for the entire D[k] window (window labeling).

    Args:
        S: np.ndarray,      (M, n_inputs) normalized stimulus vectors
        D: np.ndarray,      (M,) delay between stimuli in ESN timesteps
        SL: np.ndarray,      (M,) stimulus length in ESN timesteps
        L: np.ndarray,      (M, n_outputs) one-hot label per stimulus
        dt: float = 0.1,    timestep size


    Returns:
      U: (T_total, n_inputs) continuous input stream
      Y: (T_total, n_outputs) window labels
      onset_times: (M,) onset timestep of each stimulus in the stream
    """
    K, n_inputs = S.shape
    n_outputs = L.shape[1]

    assert D.shape == (K,)
    assert SL.shape == (K,)
    assert np.all(SL <= D), "Stimulus length SL[k] must be <= delay D[k] for all k."

    S = np.asarray(S, dtype=float)
    D = np.asarray(D, dtype=int)
    SL = np.asarray(SL, dtype=int)
    L = np.asarray(L, dtype=float)

    T_total = int(D.sum())
    U = np.zeros((T_total, n_inputs), dtype=float)
    Y = np.zeros((T_total, n_outputs), dtype=float)
    onset = np.zeros((K,), dtype=int)

    t = 0
    global_phase = 0.0

    for k in range(K):
        onset[k] = t
        Dk = int(D[k])
        SLk = int(SL[k])

        Y[t:t+Dk] = L[k]

        if phase_reset:
            local_phase0 = 0.0
        else:
            local_phase0 = global_phase

        # build time vector in "steps"
        steps = np.arange(SLk, dtype=float)
        # convert to continuous time
        tt = (steps * dt) + local_phase0

        # for each input dimension i: sin(2*pi * S[k,i] * tt)
        U[t:t+SLk] = np.sin(2*np.pi * (tt[:, None] * S[k][None, :]))

        global_phase = tt[-1] + dt if SLk > 0 else global_phase

        t += Dk

    return U, Y, onset


def build_stream_sinusoidal_window_labeled( S: np.ndarray, D: np.ndarray, L: np.ndarray, dt: float = 0.1, phase_reset: bool = True,):
    """Continuous multi-stimulus stream builder.

    Each stimulus k occupies a window of length D[k].
    Within that window, each feature drives one sinusoid channel, exactly like the original ESN.

    Args:
        S: np.ndarray,      (M, n_inputs) normalized stimulus vectors
        D: np.ndarray,      (M,) integer window lengths in ESN timesteps
        L: np.ndarray,      (M, n_outputs) one-hot label per stimulus
        dt: float = 0.1,    timestep size


    Returns:
      U: (T_total, n_inputs) continuous input stream
      Y: (T_total, n_outputs) window labels
      onset_times: (M,) onset timestep of each stimulus in the stream
    """
    S = np.asarray(S, dtype=float)
    D = np.asarray(D, dtype=int)
    L = np.asarray(L, dtype=float)

    assert S.shape[0] == D.shape[0] == L.shape[0]
    M, n_in = S.shape
    n_out = L.shape[1]

    T_total = int(D.sum())
    U = np.zeros((T_total, n_in), dtype=float)
    Y = np.zeros((T_total, n_out), dtype=float)
    onset_times = np.zeros((M,), dtype=int)

    t0 = 0
    global_phase = 0.0

    for k in range(M):
        onset_times[k] = t0
        T = int(D[k])

        # local time for window
        if phase_reset:
            t = np.arange(T) * dt
        else:
            t = (np.arange(T) * dt) + global_phase

        U_seg = np.sin((2.0 * np.pi) * t[:, None] * S[k][None, :])

        U[t0:t0+T, :] = U_seg
        Y[t0:t0+T, :] = L[k]

        t0 += T
        if not phase_reset:
            global_phase = t[-1] + dt

    return U, Y, onset_times


def accuracy_elementwise(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    yp = Y_pred.argmax(axis=1)
    yt = Y_true.argmax(axis=1)
    return (yp == yt).mean()

def build_stream_window_labeled(stimulus_matrix, delays, labels):
    """
    Build a continuous input stream U(t) and window labels Y(t).

        S: (M, n_inputs) stimuli
        D: (M,) integer delays/window lengths
        L: (M, n_outputs) one-hot labels per stimulus

    Stream semantics:
    - at time t_k, inject stimulus S[k] for ONE timestep (impulse)
    - for t in [t_k, t_k + D[k]) label is L[k]
    - between impulses input is zero, echo is internal
    """
    stimulus_matrix = np.asarray(stimulus_matrix, dtype=float)
    delays = np.asarray(delays, dtype=int)
    labels = np.asarray(labels, dtype=float)

    assert stimulus_matrix.shape[0] == delays.shape[0] == labels.shape[0] #Ensure data sizes are correct

    M, n_in = stimulus_matrix.shape
    n_out = labels.shape[1]

    T_total = int(delays.sum())
    U = np.zeros((T_total, n_in), dtype=float)
    Y = np.zeros((T_total, n_out), dtype=float)

    t = 0
    onset_times = np.zeros((M,), dtype=int)

    for k in range(M):
        onset_times[k] = t

        U[t] = stimulus_matrix[k]

        Y[t : t + delays[k]] = labels[k]

        t += delays[k]

    return U, Y, onset_times
// GCN-195M functional-dynamic model configuration.
// Reference points from NP-RC-2-036 are used where they are explicitly available.
// Simplified couplings, latent states and time constants remain model assumptions.

clear gcn;

function script_dir = gcn_get_script_dir(script_name)
    script_dir = get_absolute_file_path(script_name);
    if script_dir == "" then
        script_dir = pwd() + filesep();
    end
    if part(script_dir, length(script_dir)) == filesep() then
        script_dir = part(script_dir, 1:(length(script_dir) - 1));
    end
endfunction

function label = gcn_mode_label_from_code(code)
    labels = ["normal";
              "high_vibration";
              "bearing_overheat";
              "head_drop";
              "motor_overload"];
    idx = int(max(1, min(size(labels, "*"), code)));
    label = labels(idx);
endfunction

function series = gcn_make_series(t, values)
    series = struct("time", t, "values", values);
endfunction

function messages = gcn_add_message(messages, text)
    if size(messages, "*") == 0 then
        messages = text;
    else
        messages($ + 1) = text;
    end
endfunction

function report = gcn_validate_current_run(t, y)
    global gcn;

    ok = %t;
    messages = [];

    if size(t, "*") == 0 | size(y, "*") == 0 then
        ok = %f;
        messages = gcn_add_message(messages, "Output workspace series is empty.");
        report = struct("ok", ok, "messages", messages, "pre_mean", [], "post_mean", [], "delta_mean", []);
        return;
    end

    if or(isnan(y)) then
        ok = %f;
        messages = gcn_add_message(messages, "Detected NaN in output channels.");
    end

    if or(abs(y) == %inf) then
        ok = %f;
        messages = gcn_add_message(messages, "Detected Inf in output channels.");
    end

    pre_idx = find(t <= max(10.0, gcn.sim.t_fault - 8.0));
    if size(pre_idx, "*") == 0 then
        pre_idx = 1:max(2, int(0.2 * size(t, "*")));
    end

    post_start = min(gcn.sim.t_end - 15.0, gcn.sim.t_fault + 3.0 * max(gcn.output_tau_vector));
    post_idx = find(t >= post_start);
    if size(post_idx, "*") == 0 then
        post_idx = max(1, size(t, "*") - int(0.2 * size(t, "*"))):size(t, "*");
    end

    pre_mean = mean(y(pre_idx, :), "r")';
    post_mean = mean(y(post_idx, :), "r")';
    delta_mean = post_mean - pre_mean;

    if gcn.mode_code == gcn.mode_codes.normal then
        if or(abs(delta_mean) > gcn.validation.normal_delta_limit) then
            ok = %f;
            messages = gcn_add_message(messages, "Normal mode drift exceeds the admissible envelope.");
        end
    else
        expected_delta = gcn.mode_output_delta_matrix(gcn.mode_code, :)';
        significant_idx = find(abs(expected_delta) >= gcn.validation.min_expected_delta);

        for k = significant_idx'
            if abs(delta_mean(k)) >= gcn.validation.sign_ignore_limit(k) then
                if sign(delta_mean(k)) <> sign(expected_delta(k)) then
                    ok = %f;
                    messages = gcn_add_message(messages, msprintf("Unexpected sign in channel %s.", gcn.channels(k)));
                end
            end
        end

        scaled_delta = abs(delta_mean ./ gcn.validation.channel_scale);
        dominant_score = max(scaled_delta);

        select gcn.mode_code
        case gcn.mode_codes.high_vibration then
            if scaled_delta(1) + 1.0d-9 < dominant_score then
                ok = %f;
                messages = gcn_add_message(messages, "High-vibration mode is not dominated by V.");
            end
        case gcn.mode_codes.bearing_overheat then
            if scaled_delta(2) + 1.0d-9 < dominant_score then
                ok = %f;
                messages = gcn_add_message(messages, "Bearing-overheat mode is not dominated by Tb.");
            end
        case gcn.mode_codes.head_drop then
            if max([scaled_delta(4); scaled_delta(5)]) + 1.0d-9 < dominant_score then
                ok = %f;
                messages = gcn_add_message(messages, "Head-drop mode is not dominated by H/Q.");
            end
            if ~(delta_mean(4) < 0 & delta_mean(5) < 0) then
                ok = %f;
                messages = gcn_add_message(messages, "Head-drop mode does not decrease H and Q together.");
            end
        case gcn.mode_codes.motor_overload then
            if scaled_delta(3) + 1.0d-9 < dominant_score then
                ok = %f;
                messages = gcn_add_message(messages, "Motor-overload mode is not dominated by I.");
            end
        end

        if dominant_score < gcn.validation.min_anomaly_score(gcn.mode_code) then
            ok = %f;
            messages = gcn_add_message(messages, "Anomaly response is too weak for the selected mode.");
        end
    end

    report = struct( ..
        "ok", ok, ..
        "messages", messages, ..
        "pre_mean", pre_mean, ..
        "post_mean", post_mean, ..
        "delta_mean", delta_mean);
endfunction

global gcn;

root_dir = gcn_get_script_dir("config_gcn195m.sce");
output_dir = root_dir + filesep() + ".." + filesep() + "data_demo";

if ~isdir(output_dir) then
    mkdir(output_dir);
end

gcn = struct();
gcn.paths = struct( ..
    "root_dir", root_dir, ..
    "model_file", root_dir + filesep() + "gcn195m_model.zcos", ..
    "builder_file", root_dir + filesep() + "build_gcn195m_xcos.sce", ..
    "runner_file", root_dir + filesep() + "run_gcn195m_model.sce", ..
    "export_file", root_dir + filesep() + "export_gcn_dataset.sce", ..
    "output_dir", output_dir);

gcn.mode_codes = struct("normal", 1, ..
                        "high_vibration", 2, ..
                        "bearing_overheat", 3, ..
                        "head_drop", 4, ..
                        "motor_overload", 5);

if exists("gcn_requested_mode_code", "local") then
    requested_mode_code = gcn_requested_mode_code;
else
    requested_mode_code = gcn.mode_codes.normal;
end

gcn.mode_code = int(max(1, min(5, requested_mode_code)));
gcn.mode_label = gcn_mode_label_from_code(gcn.mode_code);

gcn.channels = ["V"; "Tb"; "I"; "H"; "Q"];
gcn.units = ["mm/s"; "degC"; "A"; "kgf/cm2"; "m3/h"];
gcn.internal_states = ["mechanical_state";
                       "bearing_state";
                       "hydraulic_state";
                       "electrical_state";
                       "thermal_state"];

// Base operating point.
gcn.base = struct( ..
    "V", 4.50, ..      // model assumption: stable operation below the 13 mm/s short-term limit
    "Tb", 58.00, ..    // model assumption: below the 65 C normal bearing temperature range
    "I", 605.00, ..    // NP-RC-2-036: current on hot water
    "H", 6.75, ..      // NP-RC-2-036: nominal head
    "Q", 20000.0);     // NP-RC-2-036: nominal flow

gcn.base_vector = [gcn.base.V;
                   gcn.base.Tb;
                   gcn.base.I;
                   gcn.base.H;
                   gcn.base.Q];

// First-order output channel dynamics.
gcn.output_tau = struct( ..
    "V", 5.0, ..
    "Tb", 18.0, ..
    "I", 6.0, ..
    "H", 4.5, ..
    "Q", 5.5);

gcn.output_tau_vector = [gcn.output_tau.V;
                         gcn.output_tau.Tb;
                         gcn.output_tau.I;
                         gcn.output_tau.H;
                         gcn.output_tau.Q];

// First-order latent state dynamics.
gcn.state_tau = struct( ..
    "mechanical_state", 7.0, ..
    "bearing_state", 11.0, ..
    "hydraulic_state", 5.0, ..
    "electrical_state", 4.0, ..
    "thermal_state", 16.0);

gcn.state_tau_vector = [gcn.state_tau.mechanical_state;
                        gcn.state_tau.bearing_state;
                        gcn.state_tau.hydraulic_state;
                        gcn.state_tau.electrical_state;
                        gcn.state_tau.thermal_state];

// Additive measurement noise.
gcn.noise_sigma = struct( ..
    "V", 0.14, ..
    "Tb", 0.20, ..
    "I", 2.80, ..
    "H", 0.04, ..
    "Q", 42.0);

gcn.noise_sigma_vector = [gcn.noise_sigma.V;
                          gcn.noise_sigma.Tb;
                          gcn.noise_sigma.I;
                          gcn.noise_sigma.H;
                          gcn.noise_sigma.Q];

gcn.sim = struct( ..
    "t_fault", 40.0, ..
    "t_end", 180.0, ..
    "dt", 0.2, ..
    "random_seed", 28042026);

// Latent state targets by mode.
// Columns: mechanical_state, bearing_state, hydraulic_state, electrical_state, thermal_state.
// Rows: normal, high_vibration, bearing_overheat, head_drop, motor_overload.
gcn.mode_state_target_matrix = [
    0.00   0.00   0.00   0.00   0.00;
    2.00   0.40   0.20   0.25   0.35;
    0.35   1.70   0.10   0.20   1.25;
    0.35   0.15   1.45   0.00   0.25;
    0.25   0.35   0.25   1.45   0.95
];

// Output coupling matrix: [V, Tb, I, H, Q] = base + C * internal_states.
gcn.output_coupling_matrix = [
      2.40    1.20     0.80    0.60    0.40;
      1.50    6.80     1.00    3.20    8.50;
     10.00    8.00   -30.00   96.00   22.00;
     -0.10   -0.05    -1.10   -0.08   -0.05;
   -150.00  -80.00 -2050.00 -120.00 -100.00
];

gcn.state_target_vector = gcn.mode_state_target_matrix(gcn.mode_code, :)';
gcn.mode_output_delta_matrix = gcn.mode_state_target_matrix * gcn.output_coupling_matrix';
gcn.output_delta_vector = gcn.mode_output_delta_matrix(gcn.mode_code, :)';
gcn.output_target_vector = gcn.base_vector + gcn.output_delta_vector;

gcn.time = (0:gcn.sim.dt:(gcn.sim.t_end - gcn.sim.dt))';
gcn.n_samples = size(gcn.time, "*");

gcn.paths.dataset_file = gcn.paths.output_dir + filesep() + "gcn_" + gcn.mode_label + ".csv";

gcn.validation = struct();
gcn.validation.normal_delta_limit = [0.60; 1.20; 12.0; 0.18; 150.0];
gcn.validation.min_expected_delta = [0.80; 2.00; 18.0; 0.15; 200.0];
gcn.validation.sign_ignore_limit = [0.30; 0.80; 10.0; 0.08; 90.0];
gcn.validation.channel_scale = [2.0; 10.0; 50.0; 0.5; 800.0];
gcn.validation.min_anomaly_score = [0.0; 0.9; 1.0; 1.2; 1.2];

gcn.assumptions = [ ..
    "The model uses five latent internal states to represent coupled mechanical, thermal, hydraulic and electrical degradation trends."; ..
    "Output deviations are formed by a linear coupling matrix applied to latent state amplitudes. The matrix preserves the direction and relative dominance of each anomaly mode."; ..
    "Latent states and measured channels are represented by first-order lag elements. This is a functional-dynamic approximation, not a detailed rotor-hydraulic CFD/FEA model."; ..
    "The high_vibration mode is tuned to stay below the 13 mm/s short-term vibration limit cited in NP-RC-2-036."; ..
    "The motor_overload mode increases current toward the documented cold-water current range without crossing the 880 A stator current limit."; ..
    "The head_drop mode represents a hydraulic efficiency loss with simultaneous reduction of head and flow and a moderate vibration increase."; ..
    "Noise terms are low-amplitude Gaussian disturbances added only to emulate sensor dispersion and to avoid idealized flat signals."
];

// Reproducible noise for all sensor channels.
rand("seed", gcn.sim.random_seed);

gcn_noise_V_ws = gcn_make_series(gcn.time, grand(gcn.n_samples, 1, "nor", 0, gcn.noise_sigma.V));
gcn_noise_Tb_ws = gcn_make_series(gcn.time, grand(gcn.n_samples, 1, "nor", 0, gcn.noise_sigma.Tb));
gcn_noise_I_ws = gcn_make_series(gcn.time, grand(gcn.n_samples, 1, "nor", 0, gcn.noise_sigma.I));
gcn_noise_H_ws = gcn_make_series(gcn.time, grand(gcn.n_samples, 1, "nor", 0, gcn.noise_sigma.H));
gcn_noise_Q_ws = gcn_make_series(gcn.time, grand(gcn.n_samples, 1, "nor", 0, gcn.noise_sigma.Q));

fault_values = zeros(gcn.n_samples, 1);
fault_values(find(gcn.time >= gcn.sim.t_fault)) = 1;
gcn_fault_activation_ws = gcn_make_series(gcn.time, fault_values);

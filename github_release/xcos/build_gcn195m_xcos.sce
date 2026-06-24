// Programmatic Xcos builder for the GCN-195M functional-dynamic anomaly model.
// Expects `gcn` to be defined by config_gcn195m.sce.

if ~exists("gcn", "local") then
    error("The gcn configuration structure is not defined. Run config_gcn195m.sce first.");
end

loadXcosLibs();

function blk = gcn_place_block(blk, orig, sz)
    blk.graphics.orig = orig;
    if argn(2) >= 3 then
        blk.graphics.sz = sz;
    end
endfunction

function blk = gcn_make_text(text, orig, sz)
    blk = TEXT_f("define");
    blk = gcn_place_block(blk, orig, sz);
    blk.graphics.exprs = [text; "2"; "1"];
endfunction

function style = gcn_make_displayed_label_style(gui_name, label_text)
    style = gui_name + ";displayedLabel=" + label_text + ";align=center";
endfunction

function blk = gcn_make_const(value, orig)
    blk = CONST_m("define");
    blk = gcn_place_block(blk, orig, [40 40]);
    blk.graphics.exprs = string(value);
    blk.model.rpar = value;
endfunction

function blk = gcn_make_gain(value, orig)
    blk = GAINBLK_f("define");
    blk = gcn_place_block(blk, orig, [46 40]);
    blk.graphics.exprs = string(value);
    blk.model.rpar = value;
endfunction

function blk = gcn_make_sum2(orig)
    blk = SUMMATION("define");
    blk = gcn_place_block(blk, orig, [40 60]);
    blk.graphics.exprs = sci2exp([1; 1]);
    blk.model.ipar = [1; 1];
endfunction

function blk = gcn_make_first_order(tau, x0, orig)
    blk = CLSS("define");
    blk = gcn_place_block(blk, orig, [80 60]);
    a = -1.0 / tau;
    b = 1.0 / tau;
    blk.graphics.exprs = [string(a); string(b); "1"; "0"; string(x0)];
    blk.model.rpar = [a; b; 1; 0];
    blk.model.state = x0;
endfunction

function blk = gcn_make_clock(period, t0, orig)
    blk = CLOCK_c("define");
    blk = gcn_place_block(blk, orig, [40 40]);
    evtdly = blk.model.rpar.objs(2);
    evtdly.graphics.exprs = [string(period); string(t0)];
    evtdly.model.rpar = [period; t0];
    evtdly.model.firing = t0;
    blk.model.rpar.objs(2) = evtdly;
endfunction

function blk = gcn_make_fromwsb(varname, orig)
    blk = FROMWSB("define");
    blk = gcn_place_block(blk, orig, [70 40]);
    inner = blk.model.rpar.objs(1);
    inner.graphics.exprs = [varname; "1"; "1"; "0"];
    inner.model.ipar = [length(ascii(varname)); ascii(varname)'; 1; 1; 0];
    blk.model.rpar.objs(1) = inner;
    blk.graphics.style = gcn_make_displayed_label_style("FROMWSB", ..
        "З робочої<BR>області<BR><font color=""orange"">[ <b>" + varname + "</b> ]</font>");
endfunction

function blk = gcn_make_tows(varname, buffer_size, orig)
    blk = TOWS_c("define");
    blk = gcn_place_block(blk, orig, [60 40]);
    blk.graphics.exprs = [string(buffer_size); varname; "0"];
    blk.model.ipar = [buffer_size; length(ascii(varname)); ascii(varname)'];
    blk.model.blocktype = "d";
    blk.graphics.style = gcn_make_displayed_label_style("TOWS_c", ..
        "До робочої<BR>області<BR><font color=""orange"">[ <b>" + varname + "</b> ]</font>");
endfunction

function blk = gcn_make_mux(n_inputs, orig)
    blk = MUX("define");
    [model, graphics, ok] = check_io(blk.model, blk.graphics, -[1:n_inputs]', 0, [], []);
    if ~ok then
        error("Failed to configure MUX block ports.");
    end
    blk.model = model;
    blk.graphics = graphics;
    blk = gcn_place_block(blk, orig, [20 100]);
    blk.graphics.exprs = string(n_inputs);
    blk.model.ipar = n_inputs;
    blk.graphics.style = gcn_make_displayed_label_style("MUX", "Мультиплексор");
endfunction

function blk = gcn_make_split(orig)
    blk = SPLIT_f("define");
    blk = gcn_place_block(blk, orig, [20 20]);
endfunction

function label = gcn_mode_label_ua(code)
    labels = ["Нормальний";
              "Підвищена вібрація";
              "Перегрів підшипника";
              "Падіння напору";
              "Перевантаження двигуна"];
    idx = int(max(1, min(size(labels, "*"), code)));
    label = labels(idx);
endfunction

function label = gcn_mode_label_ua_short(code)
    labels = ["Нормальний";
              "Висока вібрація";
              "Перегрів підшипника";
              "Падіння напору";
              "Перевантаження"];
    idx = int(max(1, min(size(labels, "*"), code)));
    label = labels(idx);
endfunction

function [objs, idx] = gcn_append(objs, obj)
    objs($ + 1) = obj;
    idx = length(objs);
endfunction

function [objs, idx] = gcn_add_explicit_link(objs, src_blk, src_port, dst_blk, dst_port)
    lnk = scicos_link(from=[src_blk src_port 0], to=[dst_blk dst_port 1]);
    objs($ + 1) = lnk;
    idx = length(objs);
endfunction

function [objs, idx] = gcn_add_event_link(objs, src_blk, dst_blk)
    lnk = scicos_link(ct=[5 -1], from=[src_blk 1 0], to=[dst_blk 1 1]);
    objs($ + 1) = lnk;
    idx = length(objs);
endfunction

props = scicos_params(tf=gcn.sim.t_end, Title="Функціонально-динамічна модель аномалій ГЦН-195М");
function text = gcn_num(value)
    text = msprintf("%.17g", value);
endfunction

function context = gcn_make_xcos_context()
    global gcn;

    context = [
        "// Self-contained workspace data for manual Xcos simulation.";
        "gcn_context_dt = " + gcn_num(gcn.sim.dt) + ";";
        "gcn_context_t_end = " + gcn_num(gcn.sim.t_end) + ";";
        "gcn_context_t_fault = " + gcn_num(gcn.sim.t_fault) + ";";
        "gcn_context_time = matrix(0:gcn_context_dt:(gcn_context_t_end - gcn_context_dt), -1, 1);";
        "gcn_context_n = size(gcn_context_time, ""*"");";
        "rand(""seed"", " + string(gcn.sim.random_seed) + ");";
        "gcn_context_fault = zeros(gcn_context_n, 1);";
        "gcn_context_fault(find(gcn_context_time >= gcn_context_t_fault)) = 1;";
        "gcn_fault_activation_ws = struct(""time"", gcn_context_time, ""values"", gcn_context_fault);";
        "gcn_noise_V_ws = struct(""time"", gcn_context_time, ""values"", grand(gcn_context_n, 1, ""nor"", 0, " + gcn_num(gcn.noise_sigma.V) + "));";
        "gcn_noise_Tb_ws = struct(""time"", gcn_context_time, ""values"", grand(gcn_context_n, 1, ""nor"", 0, " + gcn_num(gcn.noise_sigma.Tb) + "));";
        "gcn_noise_I_ws = struct(""time"", gcn_context_time, ""values"", grand(gcn_context_n, 1, ""nor"", 0, " + gcn_num(gcn.noise_sigma.I) + "));";
        "gcn_noise_H_ws = struct(""time"", gcn_context_time, ""values"", grand(gcn_context_n, 1, ""nor"", 0, " + gcn_num(gcn.noise_sigma.H) + "));";
        "gcn_noise_Q_ws = struct(""time"", gcn_context_time, ""values"", grand(gcn_context_n, 1, ""nor"", 0, " + gcn_num(gcn.noise_sigma.Q) + "));"
    ];
endfunction

props.context = gcn_make_xcos_context();
objs = list();

state_display_labels = ["Механічний стан";
                        "Стан підшипникового вузла";
                        "Гідравлічний стан";
                        "Електричне навантаження";
                        "Тепловий стан"];

channel_display_labels = ["Вібрація V";
                          "Температура Tb";
                          "Струм I";
                          "Напір H";
                          "Витрата Q"];

// Section headers.
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Вибір режиму", [20 20], [150 24]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Активація аномалії та приховані стани", [160 20], [340 24]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Формування зв’язаних аномальних впливів", [540 20], [360 24]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Базові параметри та динаміка виходів", [980 20], [360 24]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Шуми та збурення", [1410 20], [180 24]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Мультиплексування та експорт", [1650 20], [220 24]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Матриця коефіцієнтів зв’язку", [600 48], [220 20]));

[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Код режиму", [20 78], [82 20]));
[objs, idx_mode_const_vis] = gcn_append(objs, gcn_make_const(gcn.mode_code, [30 105]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Режим: " + gcn_mode_label_ua_short(gcn.mode_code), [85 110], [190 22]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text(msprintf("Момент t_fault = %.1f с", gcn.sim.t_fault), [20 165], [170 22]));

[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Активація аномалії", [180 78], [120 20]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Цільовий режим", [320 78], [100 20]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Інерційність станів", [430 78], [110 20]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Мех. стан", [536 78], [76 20]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Підшипник", [614 78], [76 20]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Гідравліка", [694 78], [82 20]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Ел. навант.", [774 78], [82 20]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Тепл. стан", [854 78], [76 20]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Базовий рівень", [950 78], [92 20]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Ланцюг сумування", [1188 78], [118 20]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Вихідна інерц. ланка", [1324 78], [126 20]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Шум", [1462 78], [44 20]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Суматор", [1548 78], [56 20]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Мультиплексор результатів", [1618 78], [138 20]));
[objs, idx_dummy] = gcn_append(objs, gcn_make_text("Годинник і запис", [1720 78], [120 20]));

state_y = [150; 290; 430; 570; 710];
output_y = state_y;
fault_name = "gcn_fault_activation_ws";
noise_names = ["gcn_noise_V_ws";
               "gcn_noise_Tb_ws";
               "gcn_noise_I_ws";
               "gcn_noise_H_ws";
               "gcn_noise_Q_ws"];
output_ws_names = ["gcn_V_ws";
                   "gcn_Tb_ws";
                   "gcn_I_ws";
                   "gcn_H_ws";
                   "gcn_Q_ws"];
coupling_x = [540; 620; 700; 780; 860];
sum_x = [1020; 1090; 1160; 1230; 1300];

[objs, idx_result_mux] = gcn_append(objs, gcn_make_mux(5, [1660 240]));
[objs, idx_mode_const_exp] = gcn_append(objs, gcn_make_const(gcn.mode_code, [1660 360]));
[objs, idx_mode_clock] = gcn_append(objs, gcn_make_clock(gcn.sim.dt, 0.0, [1740 360]));
[objs, idx_mode_tows] = gcn_append(objs, gcn_make_tows("gcn_mode_code_ws", gcn.n_samples, [1780 360]));

state_fault_idx = zeros(5, 1);
state_gain_idx = zeros(5, 1);
state_lag_idx = zeros(5, 1);
coupling_gain_idx = zeros(5, 5);
base_idx = zeros(5, 1);
chain_sum_idx = zeros(5, 5);
output_lag_idx = zeros(5, 1);
noise_idx = zeros(5, 1);
noise_sum_idx = zeros(5, 1);
output_tows_idx = zeros(5, 1);
output_clock_idx = zeros(5, 1);
state_split_idx = zeros(5, 2);
output_split_idx = zeros(5, 1);

for s = 1:5
    y_state = state_y(s);
    [objs, idx_dummy] = gcn_append(objs, gcn_make_text(state_display_labels(s), [18 y_state + 10], [160 20]));
    [objs, state_fault_idx(s)] = gcn_append(objs, gcn_make_fromwsb(fault_name, [190 y_state]));
    [objs, state_gain_idx(s)] = gcn_append(objs, gcn_make_gain(gcn.state_target_vector(s), [320 y_state]));
    [objs, state_lag_idx(s)] = gcn_append(objs, gcn_make_first_order(gcn.state_tau_vector(s), 0.0, [430 y_state - 10]));

    [objs, state_split_idx(s, 1)] = gcn_append(objs, gcn_make_split([coupling_x(s) - 30 output_y(1) + 10]));
    [objs, state_split_idx(s, 2)] = gcn_append(objs, gcn_make_split([coupling_x(s) - 30 output_y(3) + 10]));
end

for k = 1:5
    y_output = output_y(k);

    [objs, idx_dummy] = gcn_append(objs, gcn_make_text(channel_display_labels(k), [904 y_output - 28], [122 20]));

    for s = 1:5
        [objs, coupling_gain_idx(k, s)] = gcn_append(objs, gcn_make_gain(gcn.output_coupling_matrix(k, s), [coupling_x(s) y_output]));
    end

    [objs, base_idx(k)] = gcn_append(objs, gcn_make_const(gcn.base_vector(k), [960 y_output]));

    for p = 1:5
        [objs, chain_sum_idx(k, p)] = gcn_append(objs, gcn_make_sum2([sum_x(p) y_output - 10]));
    end

    [objs, output_lag_idx(k)] = gcn_append(objs, gcn_make_first_order(gcn.output_tau_vector(k), gcn.base_vector(k), [1380 y_output - 10]));
    [objs, noise_idx(k)] = gcn_append(objs, gcn_make_fromwsb(noise_names(k), [1500 y_output]));
    [objs, noise_sum_idx(k)] = gcn_append(objs, gcn_make_sum2([1600 y_output - 10]));
    [objs, output_split_idx(k)] = gcn_append(objs, gcn_make_split([1670 y_output + 10]));
    [objs, output_clock_idx(k)] = gcn_append(objs, gcn_make_clock(gcn.sim.dt, 0.0, [1740 y_output]));
    [objs, output_tows_idx(k)] = gcn_append(objs, gcn_make_tows(output_ws_names(k), gcn.n_samples, [1780 y_output]));
end

[objs, idx_dummy] = gcn_add_event_link(objs, idx_mode_clock, idx_mode_tows);
[objs, idx_dummy] = gcn_add_explicit_link(objs, idx_mode_const_exp, 1, idx_mode_tows, 1);

for s = 1:5
    [objs, idx_dummy] = gcn_add_explicit_link(objs, state_fault_idx(s), 1, state_gain_idx(s), 1);
    [objs, idx_dummy] = gcn_add_explicit_link(objs, state_gain_idx(s), 1, state_lag_idx(s), 1);
    [objs, idx_dummy] = gcn_add_explicit_link(objs, state_lag_idx(s), 1, state_split_idx(s, 1), 1);
    [objs, idx_dummy] = gcn_add_explicit_link(objs, state_split_idx(s, 1), 1, coupling_gain_idx(1, s), 1);
    [objs, idx_dummy] = gcn_add_explicit_link(objs, state_split_idx(s, 1), 2, coupling_gain_idx(2, s), 1);
    [objs, idx_dummy] = gcn_add_explicit_link(objs, state_split_idx(s, 1), 3, state_split_idx(s, 2), 1);
    [objs, idx_dummy] = gcn_add_explicit_link(objs, state_split_idx(s, 2), 1, coupling_gain_idx(3, s), 1);
    [objs, idx_dummy] = gcn_add_explicit_link(objs, state_split_idx(s, 2), 2, coupling_gain_idx(4, s), 1);
    [objs, idx_dummy] = gcn_add_explicit_link(objs, state_split_idx(s, 2), 3, coupling_gain_idx(5, s), 1);
end

for k = 1:5
    [objs, idx_dummy] = gcn_add_explicit_link(objs, base_idx(k), 1, chain_sum_idx(k, 1), 1);
    [objs, idx_dummy] = gcn_add_explicit_link(objs, coupling_gain_idx(k, 1), 1, chain_sum_idx(k, 1), 2);

    for p = 2:5
        [objs, idx_dummy] = gcn_add_explicit_link(objs, chain_sum_idx(k, p - 1), 1, chain_sum_idx(k, p), 1);
        [objs, idx_dummy] = gcn_add_explicit_link(objs, coupling_gain_idx(k, p), 1, chain_sum_idx(k, p), 2);
    end

    [objs, idx_dummy] = gcn_add_explicit_link(objs, chain_sum_idx(k, 5), 1, output_lag_idx(k), 1);
    [objs, idx_dummy] = gcn_add_explicit_link(objs, output_lag_idx(k), 1, noise_sum_idx(k), 1);
    [objs, idx_dummy] = gcn_add_explicit_link(objs, noise_idx(k), 1, noise_sum_idx(k), 2);
    [objs, idx_dummy] = gcn_add_explicit_link(objs, noise_sum_idx(k), 1, output_split_idx(k), 1);
    [objs, idx_dummy] = gcn_add_explicit_link(objs, output_split_idx(k), 1, idx_result_mux, k);
    [objs, idx_dummy] = gcn_add_explicit_link(objs, output_split_idx(k), 2, output_tows_idx(k), 1);
    [objs, idx_dummy] = gcn_add_event_link(objs, output_clock_idx(k), output_tows_idx(k));
end

scs_m = scicos_diagram(props=props, objs=objs);
xcosDiagramToScilab(gcn.paths.model_file, scs_m);

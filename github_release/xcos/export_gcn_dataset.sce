script_dir = get_absolute_file_path("export_gcn_dataset.sce");
if script_dir == "" then
    script_dir = pwd() + filesep();
end

if ~exists("gcn", "local") then
    if ~exists("gcn_requested_mode_code", "local") then
        gcn_requested_mode_code = 1;
    end
    exec(script_dir + "config_gcn195m.sce", -1);
end

if ~exists("gcn_outputs_ws", "local") then
    exec(script_dir + "build_gcn195m_xcos.sce", -1);
    importXcosDiagram(gcn.paths.model_file);
    [%cpr, ok] = xcos_simulate(scs_m, 4);
    if ~ok then
        error("Xcos simulation failed during CSV export.");
    end
end

if ~exists("gcn_outputs_ws", "local") | size(gcn_outputs_ws.time, "*") == 0 then
    ws_names = ["gcn_V_ws"; "gcn_Tb_ws"; "gcn_I_ws"; "gcn_H_ws"; "gcn_Q_ws"];
    t = [];
    y = zeros(gcn.n_samples, 5);

    for k = 1:5
        if ~exists(ws_names(k), "local") then
            error("Missing channel workspace series: " + ws_names(k));
        end

        ws_series = evstr(ws_names(k));
        if k == 1 then
            t = ws_series.time;
        end
        y(:, k) = ws_series.values;
    end

    gcn_outputs_ws = struct("time", t, "values", y);
end

t = gcn_outputs_ws.time;
y = gcn_outputs_ws.values;
n = size(t, "*");

if exists("gcn_mode_code_ws", "local") then
    mode_values = round(gcn_mode_code_ws.values);
    if size(mode_values, "*") <> n then
        mode_values = gcn.mode_code * ones(n, 1);
    end
else
    mode_values = gcn.mode_code * ones(n, 1);
end

fd = mopen(gcn.paths.dataset_file, "wt");
if fd < 0 then
    error("Unable to open CSV file for writing: " + gcn.paths.dataset_file);
end

mputl("t,mode_code,mode_label,V,Tb,I,H,Q", fd);

for k = 1:n
    label = gcn_mode_label_from_code(mode_values(k));
    line = msprintf("%.6f,%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f", ..
                    t(k), mode_values(k), label, ..
                    y(k, 1), y(k, 2), y(k, 3), y(k, 4), y(k, 5));
    mputl(line, fd);
end

mclose(fd);
mprintf("CSV exported: %s\n", gcn.paths.dataset_file);

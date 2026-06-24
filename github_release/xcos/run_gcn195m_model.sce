function lines = append_line(lines, text)
    if size(lines, "*") == 0 then
        lines = text;
    else
        lines($ + 1) = text;
    end
endfunction

funcprot(0);

start_dir = pwd();

if isfile("config_gcn195m.sce") then
    script_dir = pwd() + filesep();
elseif isfile("xcos" + filesep() + "config_gcn195m.sce") then
    cd("xcos");
    script_dir = pwd() + filesep();
else
    error("The file config_gcn195m.sce was not found in the current directory or in xcos.");
end

data_dir = script_dir + ".." + filesep() + "data_demo";
mode_codes = [1; 2; 3; 4; 5];
validation_errors = [];
generated_files = [];

for idx = 1:size(mode_codes, "*")
    gcn_requested_mode_code = mode_codes(idx);

    clear gcn gcn_outputs_ws gcn_mode_code_ws gcn_fault_activation_ws;
    clear gcn_noise_V_ws gcn_noise_Tb_ws gcn_noise_I_ws gcn_noise_H_ws gcn_noise_Q_ws;
    clear scs_m %cpr;

    exec(script_dir + "config_gcn195m.sce", -1);
    exec(script_dir + "build_gcn195m_xcos.sce", -1);

    if ~isfile(gcn.paths.model_file) then
        cd(start_dir);
        error("Xcos model file was not created: " + gcn.paths.model_file);
    end

    importXcosDiagram(gcn.paths.model_file);
    [%cpr, ok] = xcos_simulate(scs_m, 4);

    if ~ok then
        cd(start_dir);
        error("Xcos simulation failed for mode " + gcn.mode_label + ".");
    end

    if ~exists("gcn_outputs_ws", "local") | size(gcn_outputs_ws.time, "*") == 0 then
        ws_names = ["gcn_V_ws"; "gcn_Tb_ws"; "gcn_I_ws"; "gcn_H_ws"; "gcn_Q_ws"];
        t = [];
        y = zeros(gcn.n_samples, 5);

        for k = 1:5
            if ~exists(ws_names(k), "local") then
                cd(start_dir);
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

    exec(script_dir + "export_gcn_dataset.sce", -1);

    if ~isfile(gcn.paths.dataset_file) then
        cd(start_dir);
        error("CSV file was not created: " + gcn.paths.dataset_file);
    end

    report = gcn_validate_current_run(gcn_outputs_ws.time, gcn_outputs_ws.values);
    generated_files = append_line(generated_files, gcn.paths.dataset_file);

    mprintf("Mode %d [%s]\n", gcn.mode_code, gcn.mode_label);
    mprintf("  CSV: %s\n", gcn.paths.dataset_file);

    if report.ok then
        mprintf("  validation: OK\n");
    else
        mprintf("  validation: FAILED\n");
        for k = 1:size(report.messages, "*")
            mprintf("    - %s\n", report.messages(k));
            validation_errors = append_line(validation_errors, gcn.mode_label + ": " + report.messages(k));
        end
    end
end

expected_files = [data_dir + filesep() + "gcn_normal.csv";
                  data_dir + filesep() + "gcn_high_vibration.csv";
                  data_dir + filesep() + "gcn_bearing_overheat.csv";
                  data_dir + filesep() + "gcn_head_drop.csv";
                  data_dir + filesep() + "gcn_motor_overload.csv"];

for k = 1:size(expected_files, "*")
    if ~isfile(expected_files(k)) then
        validation_errors = append_line(validation_errors, "Missing CSV file: " + expected_files(k));
    end
end

if size(validation_errors, "*") > 0 then
    mprintf("\nValidation summary:\n");
    for k = 1:size(validation_errors, "*")
        mprintf("  - %s\n", validation_errors(k));
    end
    cd(start_dir);
    error("One or more validation checks failed.");
end

mprintf("\nAll modes completed successfully.\n");
for k = 1:size(generated_files, "*")
    mprintf("  %s\n", generated_files(k));
end

cd(start_dir);

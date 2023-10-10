import os
import sys
from pprint import pprint

opt_script = r"E:\miniconda\envs\paddle_test\Scripts\paddle_lite_opt"
valid_target = "x86"

if bool(0):
    cmd = f"{sys.executable} {opt_script}"
    pprint(cmd.split())
    os.system(cmd)
else:
    for i in range(1, 2 + 1):
        model_file = rf"data/export/DTLN_0831_wSDR_drb_pre70ms_none_triple_endto1.0_ep50_p{i}.pdmodel"
        param_file = rf"data/export/DTLN_0831_wSDR_drb_pre70ms_none_triple_endto1.0_ep50_p{i}.pdiparams"
        optim_out_prefix = rf"data/export_opt/DTLN_0831_wSDR_drb_pre70ms_none_triple_endto1.0_ep50_p{i}"

        cmd = f"""\
        {sys.executable} {opt_script} \
            --model_file={model_file} \
            --param_file={param_file} \
            --optimize_out_type="naive_buffer" \
            --optimize_out={optim_out_prefix} \
            --valid_targets={valid_target} \
            --quant_model=true \
            --quant_type=QUANT_INT16 \
            --enable_fp16=true
        """

        # cmd = f"""\
        # {sys.executable} {opt_script} \
        #     --model_file={model_file} \
        #     --param_file={param_file} \
        #     --optimize_out_type="naive_buffer" \
        #     --valid_targets={valid_target} \
        #     --print_model_ops=true
        # """

        pprint(cmd.split())
        os.system(cmd)

        # opt = lite.Opt()
        # opt.set_model_file(model_file)
        # opt.set_param_file(param_file)
        # opt.set_optimize_out(optim_out_prefix)
        # opt.set_optimize_out("naive_buffer")
        # opt.set_quant_model(True)
        # opt.set_quant_type("QUANT_INT16")
        # opt.set_valid_places(valid_target)
        # opt.enable_fp16()
        # opt.run()

"""Stage-output OOF for a given base OOF (22.09): the E74 stage (accepted 1 kt extraction) fitted per fold on top
of E36_BASE_DIR, cand arm only, written to STAGE_OUT. This is the reference blend weights must be fitted against
(v91 lesson): the object that goes to the board, not the bare base.
    E36_BASE_DIR=data/cv_preds_e89_cand STAGE_OUT=data/cv_preds_e89_stage .venv/bin/python src/stage_oof.py
"""
import os
import re

out = os.environ["STAGE_OUT"]
src = open(os.path.join(os.path.dirname(__file__), "e74_onset_plus_stack.py")).read()
src = src.split("\ndec, tot, small")[0]
src = re.sub(r'arms = \{"ctrl": \(frame\("data/adsblol", "data/adsblol_standwalk"\), CTRL_DIR\),\n\s*"cand":',
             'arms = {"cand":', src)
src = src.replace('CAND_DIR = "data/cv_preds_e74_ctrl", "data/cv_preds_e74"', f'CAND_DIR = None, "{out}"')
assert '"ctrl": (frame' not in src and out in src, "patch failed"
exec(compile(src, "e74_onset_plus_stack.py", "exec"), {"__name__": "__main__", "__file__": os.path.join(os.path.dirname(__file__), "e74_onset_plus_stack.py")})

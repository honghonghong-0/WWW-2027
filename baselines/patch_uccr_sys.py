"""WWW27 audit fixes for UCCR system: (1) restore best-valid model + recompute train memory before final test,
(2) build train memory under no_grad (same values, no autograd graph), (3) skip per-epoch test (selection uses valid only)."""
import shutil, os
p = "/path/to/base/projects/crs/external/UCCR_infer/crslab/system/uccr.py"
if not os.path.exists(p + ".bak_www27audit"): shutil.copy(p, p + ".bak_www27audit")
s = open(p).read()
a = "                for batch_all_tr in self.train_dataloader.get_rec_data(self.rec_batch_size, shuffle=True):\n"
assert s.count(a) == 1
s = s.replace(a, "              with torch.no_grad():  # WWW27 audit: memory reps are detached anyway\n" + a.replace("                for", "                for", 1))
# indent the body of that for-loop block by nothing: python allows 'with' at col 14 and 'for' at col 16 -> valid nesting
b = """                ################################### Test!
                self.evaluator.reset_metrics()
                for batch in self.test_dataloader.get_rec_data(self.rec_batch_size*2, shuffle=False):
                    self.step(batch, stage='rec', mode='test')
                self.evaluator.report()
"""
assert s.count(b) == 1
s = s.replace(b, """                ################################### Test! (WWW27 audit: disabled; model selection uses valid only)
""")
c = """        www27_hook.STATE["epoch"] = "final"
        logger.info('[Test]')
"""
assert s.count(c) == 1
s = s.replace(c, """        www27_hook.STATE["epoch"] = "final"
        # WWW27 audit: restore best-valid weights (saved by early_stop) and rebuild train memory with them
        if os.path.exists(self.model_file):
            self.restore_model(); logger.info('[WWW27 restored best-valid model]')
        self.tr_his_words_reps = []; self.tr_his_entities_reps = []; self.tr_word_attn_rep = []; self.tr_entity_attn_rep = []; self.tr_user_id = []
        logger.info('[Test]')
""")
if "\nimport os\n" not in s: s = "import os\n" + s
open(p, "w").write(s)
import ast; ast.parse(s); print("OK")

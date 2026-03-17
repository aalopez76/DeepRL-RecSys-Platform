import json
import unittest
from pathlib import Path
from deeprl_recsys.training.callbacks import OPEEvaluationCallback

class TestOPECallback(unittest.TestCase):
    def test_callback_writes_file(self):
        output_path = "tmp_ope_test.jsonl"
        Path(output_path).unlink(missing_ok=True)
        
        callback = OPEEvaluationCallback(eval_data={}, agent=None, interval=1, output_path=output_path)
        callback.on_train_begin()
        callback.on_step_end(1, {"reward": 0.5})
        
        self.assertTrue(Path(output_path).exists())
        with open(output_path, "r") as f:
            line = f.readline()
            data = json.loads(line)
            self.assertEqual(data["step"], 1)
            self.assertIn("ips", data)
            
        Path(output_path).unlink()

if __name__ == "__main__":
    unittest.main()

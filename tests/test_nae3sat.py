import os
import subprocess
import sys
import unittest

# /path/to/demos/nae3sat/tests/test_nae3sat.py
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestDemo(unittest.TestCase):
    @unittest.skipIf(os.getenv('SKIP_INT_TESTS'), "Skipping integration test.")
    def test_nae3sat(self):
        demo_file = os.path.join(project_dir, 'nae3sat_example.py')
        output = subprocess.check_output([sys.executable, demo_file])
        output = str(output).upper()
        if os.getenv('DEBUG_OUTPUT'):
            print("Example output \n" + output)

        for solver_name in "Adv", "Adv2_proto":
            with self.subTest(msg=f"Verify if output contains 'minor embedding problem into {solver_name}' \n"):
                self.assertIn(f'minor embedding problem into {solver_name}'.upper(), output)
            with self.subTest(msg=f"Verify if output contains 'sending problem to {solver_name}' \n"):
                self.assertIn(f'sending problem to {solver_name}'.upper(), output)    
            with self.subTest(msg="Verify if error string contains in output \n"):
                self.assertNotIn("ERROR", output)
            with self.subTest(msg="Verify if warning string contains in output \n"):
                self.assertNotIn("WARNING", output)

if __name__ == '__main__':
    unittest.main()
# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import unittest


class TestImport(unittest.TestCase):
    def test_import(self):
        import jury

        print(jury.__version__)


if __name__ == "__main__":
    unittest.main()

import unittest


class TestImport(unittest.TestCase):
    def test_import(self):
        import jury

        print(jury.__version__)


if __name__ == "__main__":
    unittest.main()

from __future__ import absolute_import

import io
import numpy as np
import torch
import unittest

try:
    import unittest.mock as mock
except ImportError:
    import mock

from laia.utils import kaldi


class TestWriteBinaryMatrix(unittest.TestCase):
    def _run_test(self, dtype):
        f = io.BytesIO()
        x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=dtype)
        kaldi.write_binary_matrix(f, x)
        # Expected code
        header = b"FM " if dtype == torch.float32 else b"DM "
        # Expected rows
        header = header + b"\x04" + b"\x02\x00\x00\x00"
        # Expected columns
        header = header + b"\x04" + b"\x04\x00\x00\x00"
        self.assertEqual(f.getvalue(), header + x.numpy().tobytes())

    def test_f32(self):
        self._run_test(torch.float32)

    def test_f64(self):
        self._run_test(torch.float64)


class TestWriteTextLattice(unittest.TestCase):
    def _run_test(self, dtype):
        f = io.StringIO()
        x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=dtype)
        kaldi.write_text_lattice(f, x, digits=3)
        expected_value = (
            "0\t1\t1\t1\t0,1.0\n"
            "0\t1\t2\t2\t0,2.0\n"
            "0\t1\t3\t3\t0,3.0\n"
            "0\t1\t4\t4\t0,4.0\n"
            "1\t2\t1\t1\t0,5.0\n"
            "1\t2\t2\t2\t0,6.0\n"
            "1\t2\t3\t3\t0,7.0\n"
            "1\t2\t4\t4\t0,8.0\n"
            "2\t0,0\n\n"
        )
        self.assertEqual(f.getvalue(), expected_value)

    def test_f32(self):
        self._run_test(torch.float32)

    def test_f64(self):
        self._run_test(torch.float64)


class TestArchiveMatrixWriter(unittest.TestCase):
    def test_write(self):
        f = io.BytesIO()
        x1 = torch.rand(7, 9, dtype=torch.float32)
        x2 = torch.rand(8, 8, dtype=torch.float64)
        writer = kaldi.ArchiveMatrixWriter(f)
        writer.write("key1", x1)
        writer.write("longerkey", x2)
        # Test written data
        binary_data = f.getvalue()
        expected_size1 = 4 + 3 + 3 + 5 + 5 + x1.numel() * x1.element_size()
        expected_size2 = 9 + 3 + 3 + 5 + 5 + x2.numel() * x2.element_size()
        self.assertEqual(len(binary_data), expected_size1 + expected_size2)
        self.assertEqual(binary_data[:7], b"key1 \x00B")
        self.assertEqual(
            binary_data[expected_size1 : (expected_size1 + 12)], b"longerkey \x00B"
        )

    def test_write_iterable(self):
        x1 = torch.rand(7, 9, dtype=torch.float32)
        x2 = torch.rand(8, 8, dtype=torch.float64)

        file1 = io.BytesIO()
        writer1 = kaldi.ArchiveMatrixWriter(file1)
        writer1.write("key1", x1)
        writer1.write("key2", x2)

        file2 = io.BytesIO()
        writer2 = kaldi.ArchiveMatrixWriter(file2)
        writer2.write_iterable(zip(["key1", "key2"], [x1, x2]))

        self.assertEqual(file2.getvalue(), file1.getvalue())


class TestRotatingArchiveMatrixWriter(unittest.TestCase):
    @mock.patch("laia.utils.kaldi.io.open")
    def test_single_file(self, mock_file_open):
        f = "./foo/bar.ark"
        mock_file_open.side_effect = [io.BytesIO(), io.StringIO()]
        x1 = torch.rand(7, 9, dtype=torch.float32)
        x2 = torch.rand(8, 8, dtype=torch.float64)
        writer = kaldi.RotatingArchiveMatrixWriter(f, maxsamples=None)
        writer.write("key1", x1)
        writer.write("longerkey", x2)
        # Test written data
        binary_data = writer._file.getvalue()
        expected_size1 = 4 + 3 + 3 + 5 + 5 + x1.numel() * x1.element_size()
        expected_size2 = 9 + 3 + 3 + 5 + 5 + x2.numel() * x2.element_size()
        self.assertEqual(len(binary_data), expected_size1 + expected_size2)
        self.assertEqual(binary_data[:7], b"key1 \x00B")
        self.assertEqual(
            binary_data[expected_size1 : (expected_size1 + 12)], b"longerkey \x00B"
        )

    @mock.patch("laia.utils.kaldi.io.open")
    def test_rotate(self, mock_file_open):
        f = "./foo/bar.ark"
        mock_file_open.side_effect = [
            io.BytesIO(),
            io.StringIO(),
            io.BytesIO(),
            io.StringIO(),
        ]
        x1 = torch.rand(7, 9, dtype=torch.float32)
        x2 = torch.rand(8, 8, dtype=torch.float64)
        writer = kaldi.RotatingArchiveMatrixWriter(f, maxsamples=1)
        writer.write("key1", x1)
        # Test written data on first file
        binary_data = writer._file.getvalue()
        expected_size1 = 4 + 3 + 3 + 5 + 5 + x1.numel() * x1.element_size()
        self.assertEqual(len(binary_data), expected_size1)
        self.assertEqual(binary_data[:7], b"key1 \x00B")

        writer.write("longerkey", x2)
        # Test written data on second file
        binary_data = writer._file.getvalue()
        expected_size2 = 9 + 3 + 3 + 5 + 5 + x2.numel() * x2.element_size()
        self.assertEqual(len(binary_data), expected_size2)
        self.assertEqual(binary_data[:12], b"longerkey \x00B")
        # Test number of files
        self.assertEqual(1, writer._file_id)

    @mock.patch("laia.utils.kaldi.io.open")
    def test_write_iterable(self, mock_file_open):
        f = "./foo/bar.ark"
        x1 = torch.rand(7, 9, dtype=torch.float32)
        x2 = torch.rand(8, 8, dtype=torch.float64)

        mock_file_open.side_effect = [
            io.BytesIO(),
            io.StringIO(),
            io.BytesIO(),
            io.StringIO(),
        ]
        writer1 = kaldi.RotatingArchiveMatrixWriter(f, maxsamples=None)
        writer1.write("key1", x1)
        writer1.write("key2", x2)

        writer2 = kaldi.RotatingArchiveMatrixWriter(f, maxsamples=None)
        writer2.write_iterable(zip(["key1", "key2"], [x1, x2]))

        self.assertEqual(writer1._file.getvalue(), writer2._file.getvalue())
        self.assertEqual(writer1._scp_file.getvalue(), writer2._scp_file.getvalue())


if __name__ == "__main__":
    unittest.main()

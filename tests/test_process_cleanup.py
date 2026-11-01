# SPDX-License-Identifier: GPL-3.0-only
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools import run_isolated_bench as runner


class CleanupTests(unittest.TestCase):
    def test_windows_native_output_is_not_decoded_as_utf8(self):
        server = Mock(pid=12345)
        server.poll.return_value = None
        result = subprocess.CompletedProcess([], 0, stderr=b"\xb3\xc9\xb9\xa6")
        with patch.object(runner, "os", SimpleNamespace(name="nt")), patch.object(runner.subprocess, "run", return_value=result) as run:
            runner.stop_owned(server)
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["taskkill", "/PID", "12345", "/T", "/F"])
        self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], subprocess.PIPE)
        self.assertNotIn("text", kwargs)
        server.wait.assert_called_once_with(timeout=30)


if __name__ == "__main__":
    unittest.main()

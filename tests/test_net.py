"""Network resolution order for the Gemini client (Oct 12 2026: IPv6 to Google was blackholed on Harvard Wi-Fi)."""
import socket
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain import llm  # noqa: E402


class PreferIPv4Tests(unittest.TestCase):
    def setUp(self):
        self.saved = socket.getaddrinfo
        self.f = self.fake                                  # one bound-method object for identity checks

    def tearDown(self):
        socket.getaddrinfo = self.saved

    def fake(self, *args, **kwargs):
        return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2607:f8b0::1", 443, 0, 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("142.250.0.1", 443)),
                (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2607:f8b0::2", 443, 0, 0))]

    def test_ipv4_comes_first_and_ipv6_stays_as_fallback(self):
        socket.getaddrinfo = self.f
        llm.prefer_ipv4()
        fams = [r[0] for r in socket.getaddrinfo("generativelanguage.googleapis.com", 443)]
        self.assertEqual(fams, [socket.AF_INET, socket.AF_INET6, socket.AF_INET6])

    def test_applied_once_and_opt_out(self):
        socket.getaddrinfo = self.f
        llm.prefer_ipv4()
        wrapped = socket.getaddrinfo
        llm.prefer_ipv4()                                   # no double wrap
        self.assertIs(socket.getaddrinfo, wrapped)
        import os
        socket.getaddrinfo = self.f
        os.environ["BRAIN_IPV6_FIRST"] = "1"
        try:
            llm.prefer_ipv4()
            self.assertIs(socket.getaddrinfo, self.f)       # system order kept
        finally:
            del os.environ["BRAIN_IPV6_FIRST"]


if __name__ == "__main__":
    unittest.main()

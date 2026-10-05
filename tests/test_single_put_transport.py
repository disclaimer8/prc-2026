import io
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from e185_single_put_upload import OnePutHTTP
from minio import Minio


class TransportTest(unittest.TestCase):
    def test_failed_and_redirected_put_are_never_retried(self):
        for status in (500, 503, 307):
            with self.subTest(status=status):
                seen = []

                class Handler(BaseHTTPRequestHandler):
                    def do_PUT(self):
                        self.rfile.read(int(self.headers['Content-Length']))
                        seen.append(self.path)
                        self.send_response(status)
                        self.send_header('Content-Length', '0')
                        self.send_header('Location', '/bucket/redirected')
                        self.end_headers()

                    def log_message(self, *args):
                        pass

                server = HTTPServer(('127.0.0.1', 0), Handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                http = OnePutHTTP()
                client = Minio(f'127.0.0.1:{server.server_port}', access_key='synthetic',
                               secret_key='synthetic', region='us-east-1', secure=False,
                               http_client=http)
                try:
                    with self.assertRaises(Exception):
                        client.put_object('bucket', 'candidate', io.BytesIO(b'hello'), 5)
                    self.assertEqual(seen, ['/bucket/candidate'])
                    with self.assertRaises(RuntimeError):
                        client.put_object('bucket', 'candidate', io.BytesIO(b'hello'), 5)
                    self.assertEqual(len(seen), 1)
                finally:
                    http.clear()
                    server.shutdown()
                    server.server_close()
                    thread.join()


if __name__ == '__main__':
    unittest.main()

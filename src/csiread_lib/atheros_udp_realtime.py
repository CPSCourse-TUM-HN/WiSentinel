from collections import deque
import logging
import socket
import threading

import numpy as np

class AtherosUDPRealtime:
    def __init__(
        self,
        port=8000,
        addr="0.0.0.0",
        nrxnum=3,
        ntxnum=3,
        tones=56,
        max_packets=200,
        log_level=logging.INFO,
    ):
        """
        Realtime UDP CSI Receiver for Atheros firmware.
        """
        self.port = port
        self.addr = addr
        self.nrxnum = nrxnum
        self.ntxnum = ntxnum
        self.tones = tones
        self.max_packets = max_packets

        # Configure logger
        self.logger = logging.getLogger(f"AtherosUDP[{self.port}]")
        self.logger.setLevel(log_level)

        # Setup UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.addr, self.port))
        self.logger.info(f"Listening for CSI packets on udp://{self.addr}:{self.port}")

        # Ring buffer for parsed packets
        self.buffer = deque(maxlen=max_packets)
        self.lock = threading.Lock()  # for thread-safe read operations

        # Runtime state
        self.packet_counter = 0
        self.success_counter = 0
        self.failure_counter = 0
        self.running = False
        self.thread = threading.Thread(target=self._recv_loop, daemon=True)

    def start(self):
        """Starts the background listener thread."""
        self.running = True
        self.thread.start()
        self.logger.info("Started background UDP listener thread")

    def stop(self):
        self.running = False
        self.sock.close()
        stats = self.get_stats()
        self.logger.info(
            f"Stopped UDP listener thread. "
            f"Packets received: {stats['packet_counter']}, "
            f"Success: {stats['success_counter']}, Failures: {stats['failure_counter']}, "
            f"Buffer left: {stats['buffer_len']}"
        )

    def get_stats(self):
        """Returns internal counters for diagnostics."""
        return {
            "packet_counter": self.packet_counter,
            "success_counter": self.success_counter,
            "failure_counter": self.failure_counter,
            "buffer_len": len(self.buffer),
        }

    def _signbit_convert(self, data, maxbit):
        if data & (1 << (maxbit - 1)):
            data -= 1 << maxbit
        return data

    def _extract_field(self, buf, curren_data, bits_left, idx, mask, field_len):
        """Extracts a signed integer field from bitstream buffer."""
        while bits_left < field_len:
            h_data = buf[idx] + (buf[idx + 1] << 8)
            idx += 2
            curren_data += h_data << bits_left
            bits_left += 16

        value = curren_data & mask
        value = self._signbit_convert(value, field_len)

        curren_data >>= field_len
        bits_left -= field_len

        return value, curren_data, bits_left, idx

    def _read_csi(self, csi_buf):
        """Parses CSI buffer from binary UDP payload."""
        try:
            nr, nc, num_tones = self.nrxnum, self.ntxnum, self.tones
            BITS_PER_FIELD = 10
            FIELD_MASK = (1 << BITS_PER_FIELD) - 1
            INIT_BITS_LEFT = 16

            csi = np.zeros((num_tones, nr, nc), dtype=np.complex128)

            idx = 0
            h_data = csi_buf[idx] + (csi_buf[idx + 1] << 8)
            idx += 2
            curren_data = h_data & 0xFFFF
            bits_left = INIT_BITS_LEFT

            for k in range(num_tones):
                for tx in range(nc):
                    for rx in range(nr):
                        imag, curren_data, bits_left, idx = self._extract_field(
                            csi_buf,
                            curren_data,
                            bits_left,
                            idx,
                            FIELD_MASK,
                            BITS_PER_FIELD,
                        )
                        real, curren_data, bits_left, idx = self._extract_field(
                            csi_buf,
                            curren_data,
                            bits_left,
                            idx,
                            FIELD_MASK,
                            BITS_PER_FIELD,
                        )
                        csi[k, rx, tx] = real + 1j * imag

            self.logger.debug(f"Parsed CSI shape: {csi.shape}")
            return csi

        except Exception as e:
            self.logger.error(f"CSI decode error ({type(e).__name__}): {e}")
            return None

    def _recv_loop(self):
        """Background thread: receive UDP packets and parse CSI."""
        while self.running:
            try:
                data, addr = self.sock.recvfrom(2048)
                self.packet_counter += 1
                csi = self._read_csi(data)
                if csi is not None and np.isnan(csi).any() == False:
                    self.success_counter += 1
                    self.buffer.append(csi)
                    if self.success_counter % 100 == 0:
                        self.logger.debug(
                            f"Buffered packets: {len(self.buffer)} | Total: {self.packet_counter}"
                        )
                else:
                    self.failure_counter += 1
                    if self.failure_counter % 50 == 0:
                        self.logger.warning(f"Failed to parse valid CSI from packet №{self.packet_counter}")
            except Exception as e:
                self.logger.warning(f"UDP receive error ({type(e).__name__}): {e}")

    def drain_buffer(self, max_packets=None):
        """
        Atomically extract multiple packets (up to `max_packets`) from buffer.
        Called by slow consumers to avoid locking per packet.
        """
        with self.lock:
            count = (
                len(self.buffer)
                if max_packets is None
                else min(len(self.buffer), max_packets)
            )
            packets = [self.buffer.popleft() for _ in range(count)]
            self.logger.debug(f"Drained {len(packets)} packets from buffer")

            stats = self.get_stats()
            self.logger.debug(
                f"[Drain] Retrieved {len(packets)} packets | "
                f"Buffer size after drain: {stats['buffer_len']} | "
                f"Total received: {stats['packet_counter']} | "
                f"Success: {stats['success_counter']} | "
                f"Fail: {stats['failure_counter']}"
            )
            if not packets:
                self.logger.debug("No packets to drain")

            return packets

    def __len__(self):
        """Safe length check with lock (not strictly necessary for deque, but consistent)."""
        with self.lock:
            return len(self.buffer)

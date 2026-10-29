"""
scenario_io.py — read Waymo Open Motion Scenario TFRecords without TensorFlow
or the waymo-open-dataset package.

TFRecord framing: uint64 length | uint32 masked CRC of length | payload |
uint32 masked CRC of payload (little endian; CRCs not checked here).

Scenario is a trimmed copy of waymo_open_dataset/protos/scenario.proto
(v1.3.0, same field numbers), built at import time from a descriptor: it keeps
the tracks and prediction targets; map, traffic-light, lidar and camera fields
are absent, so the parser skips them as unknown fields.
"""

from __future__ import annotations

import struct
from typing import Iterator

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

TYPE_NAMES = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}


def _build():
    F = descriptor_pb2.FieldDescriptorProto
    fdp = descriptor_pb2.FileDescriptorProto(name="mml_womd_scenario.proto", package="mml_womd", syntax="proto2")

    def msg(name, fields):
        m = fdp.message_type.add(name=name)
        for fname, num, ftype, label, type_name in fields:
            f = m.field.add(name=fname, number=num, type=ftype, label=label)
            if type_name:
                f.type_name = type_name
        return m

    opt, rep = F.LABEL_OPTIONAL, F.LABEL_REPEATED
    msg("ObjectState", [("center_x", 2, F.TYPE_DOUBLE, opt, None), ("center_y", 3, F.TYPE_DOUBLE, opt, None),
                        ("center_z", 4, F.TYPE_DOUBLE, opt, None), ("length", 5, F.TYPE_FLOAT, opt, None),
                        ("width", 6, F.TYPE_FLOAT, opt, None), ("height", 7, F.TYPE_FLOAT, opt, None),
                        ("heading", 8, F.TYPE_FLOAT, opt, None), ("velocity_x", 9, F.TYPE_FLOAT, opt, None),
                        ("velocity_y", 10, F.TYPE_FLOAT, opt, None), ("valid", 11, F.TYPE_BOOL, opt, None)])
    msg("Track", [("id", 1, F.TYPE_INT32, opt, None), ("object_type", 2, F.TYPE_INT32, opt, None),
                  ("states", 3, F.TYPE_MESSAGE, rep, ".mml_womd.ObjectState")])
    msg("RequiredPrediction", [("track_index", 1, F.TYPE_INT32, opt, None),
                               ("difficulty", 2, F.TYPE_INT32, opt, None)])
    msg("Scenario", [("timestamps_seconds", 1, F.TYPE_DOUBLE, rep, None),
                     ("tracks", 2, F.TYPE_MESSAGE, rep, ".mml_womd.Track"),
                     ("objects_of_interest", 4, F.TYPE_INT32, rep, None),
                     ("scenario_id", 5, F.TYPE_STRING, opt, None),
                     ("sdc_track_index", 6, F.TYPE_INT32, opt, None),
                     ("current_time_index", 10, F.TYPE_INT32, opt, None),
                     ("tracks_to_predict", 11, F.TYPE_MESSAGE, rep, ".mml_womd.RequiredPrediction")])
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("mml_womd.Scenario"))


Scenario = _build()


def tfrecords(path: str) -> Iterator[bytes]:
    with open(path, "rb") as f:
        while True:
            head = f.read(12)
            if not head:
                return
            if len(head) < 12:
                raise IOError(f"truncated record header in {path}")
            (n,) = struct.unpack("<Q", head[:8])
            payload = f.read(n)
            if len(payload) < n or len(f.read(4)) < 4:
                raise IOError(f"truncated record in {path}")
            yield payload


def scenarios(path: str) -> Iterator:
    for rec in tfrecords(path):
        s = Scenario()
        s.ParseFromString(rec)
        yield s

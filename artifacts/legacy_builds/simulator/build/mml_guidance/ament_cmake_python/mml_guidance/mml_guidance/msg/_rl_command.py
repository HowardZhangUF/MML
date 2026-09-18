# generated from rosidl_generator_py/resource/_idl.py.em
# with input from mml_guidance:msg/RLCommand.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import math  # noqa: E402, I100

# Member 'agent_state'
# Member 'target_state'
import numpy  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_RLCommand(type):
    """Metaclass of message 'RLCommand'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('mml_guidance')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'mml_guidance.msg.RLCommand')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__rl_command
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__rl_command
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__rl_command
            cls._TYPE_SUPPORT = module.type_support_msg__msg__rl_command
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__rl_command

            from std_msgs.msg import Header
            if Header.__class__._TYPE_SUPPORT is None:
                Header.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class RLCommand(metaclass=Metaclass_RLCommand):
    """Message class 'RLCommand'."""

    __slots__ = [
        '_header',
        '_control_mode',
        '_vx',
        '_vy',
        '_vz',
        '_omega',
        '_target_x',
        '_target_y',
        '_target_z',
        '_tracking_error',
        '_dt',
        '_agent_state',
        '_target_state',
    ]

    _fields_and_field_types = {
        'header': 'std_msgs/Header',
        'control_mode': 'string',
        'vx': 'float',
        'vy': 'float',
        'vz': 'float',
        'omega': 'float',
        'target_x': 'float',
        'target_y': 'float',
        'target_z': 'float',
        'tracking_error': 'float',
        'dt': 'float',
        'agent_state': 'float[4]',
        'target_state': 'float[4]',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['std_msgs', 'msg'], 'Header'),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.BasicType('float'),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('float'), 4),  # noqa: E501
        rosidl_parser.definition.Array(rosidl_parser.definition.BasicType('float'), 4),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from std_msgs.msg import Header
        self.header = kwargs.get('header', Header())
        self.control_mode = kwargs.get('control_mode', str())
        self.vx = kwargs.get('vx', float())
        self.vy = kwargs.get('vy', float())
        self.vz = kwargs.get('vz', float())
        self.omega = kwargs.get('omega', float())
        self.target_x = kwargs.get('target_x', float())
        self.target_y = kwargs.get('target_y', float())
        self.target_z = kwargs.get('target_z', float())
        self.tracking_error = kwargs.get('tracking_error', float())
        self.dt = kwargs.get('dt', float())
        if 'agent_state' not in kwargs:
            self.agent_state = numpy.zeros(4, dtype=numpy.float32)
        else:
            self.agent_state = kwargs.get('agent_state')
        if 'target_state' not in kwargs:
            self.target_state = numpy.zeros(4, dtype=numpy.float32)
        else:
            self.target_state = kwargs.get('target_state')

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.header != other.header:
            return False
        if self.control_mode != other.control_mode:
            return False
        if self.vx != other.vx:
            return False
        if self.vy != other.vy:
            return False
        if self.vz != other.vz:
            return False
        if self.omega != other.omega:
            return False
        if self.target_x != other.target_x:
            return False
        if self.target_y != other.target_y:
            return False
        if self.target_z != other.target_z:
            return False
        if self.tracking_error != other.tracking_error:
            return False
        if self.dt != other.dt:
            return False
        if any(self.agent_state != other.agent_state):
            return False
        if any(self.target_state != other.target_state):
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def header(self):
        """Message field 'header'."""
        return self._header

    @header.setter
    def header(self, value):
        if __debug__:
            from std_msgs.msg import Header
            assert \
                isinstance(value, Header), \
                "The 'header' field must be a sub message of type 'Header'"
        self._header = value

    @builtins.property
    def control_mode(self):
        """Message field 'control_mode'."""
        return self._control_mode

    @control_mode.setter
    def control_mode(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'control_mode' field must be of type 'str'"
        self._control_mode = value

    @builtins.property
    def vx(self):
        """Message field 'vx'."""
        return self._vx

    @vx.setter
    def vx(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'vx' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'vx' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._vx = value

    @builtins.property
    def vy(self):
        """Message field 'vy'."""
        return self._vy

    @vy.setter
    def vy(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'vy' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'vy' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._vy = value

    @builtins.property
    def vz(self):
        """Message field 'vz'."""
        return self._vz

    @vz.setter
    def vz(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'vz' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'vz' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._vz = value

    @builtins.property
    def omega(self):
        """Message field 'omega'."""
        return self._omega

    @omega.setter
    def omega(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'omega' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'omega' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._omega = value

    @builtins.property
    def target_x(self):
        """Message field 'target_x'."""
        return self._target_x

    @target_x.setter
    def target_x(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'target_x' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'target_x' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._target_x = value

    @builtins.property
    def target_y(self):
        """Message field 'target_y'."""
        return self._target_y

    @target_y.setter
    def target_y(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'target_y' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'target_y' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._target_y = value

    @builtins.property
    def target_z(self):
        """Message field 'target_z'."""
        return self._target_z

    @target_z.setter
    def target_z(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'target_z' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'target_z' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._target_z = value

    @builtins.property
    def tracking_error(self):
        """Message field 'tracking_error'."""
        return self._tracking_error

    @tracking_error.setter
    def tracking_error(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'tracking_error' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'tracking_error' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._tracking_error = value

    @builtins.property
    def dt(self):
        """Message field 'dt'."""
        return self._dt

    @dt.setter
    def dt(self, value):
        if __debug__:
            assert \
                isinstance(value, float), \
                "The 'dt' field must be of type 'float'"
            assert not (value < -3.402823466e+38 or value > 3.402823466e+38) or math.isinf(value), \
                "The 'dt' field must be a float in [-3.402823466e+38, 3.402823466e+38]"
        self._dt = value

    @builtins.property
    def agent_state(self):
        """Message field 'agent_state'."""
        return self._agent_state

    @agent_state.setter
    def agent_state(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float32, \
                "The 'agent_state' numpy.ndarray() must have the dtype of 'numpy.float32'"
            assert value.size == 4, \
                "The 'agent_state' numpy.ndarray() must have a size of 4"
            self._agent_state = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 4 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'agent_state' field must be a set or sequence with length 4 and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._agent_state = numpy.array(value, dtype=numpy.float32)

    @builtins.property
    def target_state(self):
        """Message field 'target_state'."""
        return self._target_state

    @target_state.setter
    def target_state(self, value):
        if isinstance(value, numpy.ndarray):
            assert value.dtype == numpy.float32, \
                "The 'target_state' numpy.ndarray() must have the dtype of 'numpy.float32'"
            assert value.size == 4, \
                "The 'target_state' numpy.ndarray() must have a size of 4"
            self._target_state = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 len(value) == 4 and
                 all(isinstance(v, float) for v in value) and
                 all(not (val < -3.402823466e+38 or val > 3.402823466e+38) or math.isinf(val) for val in value)), \
                "The 'target_state' field must be a set or sequence with length 4 and each value of type 'float' and each float in [-340282346600000016151267322115014000640.000000, 340282346600000016151267322115014000640.000000]"
        self._target_state = numpy.array(value, dtype=numpy.float32)

'''
Author: Qi7
Date: 2023-01-17 15:46:04
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-01-17 15:48:04
Description: 
'''
# required imports
from core.emulator.coreemu import CoreEmu
from core.emulator.data import IpPrefixes
from core.emulator.enumerations import EventTypes
from core.nodes.base import CoreNode, Position

# ip generator for example
ip_prefixes = IpPrefixes(ip4_prefix="10.0.0.0/24")

# create emulator instance for creating sessions and utility methods
coreemu = CoreEmu()
session = coreemu.create_session()

# must be in configuration state for nodes to start, when using "node_add" below
session.set_state(EventTypes.CONFIGURATION_STATE)

# create nodes
position = Position(x=100, y=100)
n1 = session.add_node(CoreNode, position=position)
position = Position(x=300, y=100)
n2 = session.add_node(CoreNode, position=position)

# link nodes together
iface1 = ip_prefixes.create_iface(n1)
iface2 = ip_prefixes.create_iface(n2)
session.add_link(n1.id, n2.id, iface1, iface2)

# start session
session.instantiate()

# do whatever you like here
input("press enter to shutdown")

# stop session
session.shutdown()
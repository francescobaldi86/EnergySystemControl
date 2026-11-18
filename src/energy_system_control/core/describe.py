"""Human- and machine-readable description of an Environment's topology.

`build_graph(env)` walks the environment's components, ports, connections,
sensors and controllers and returns a plain dict. `render_text` /
`render_mermaid` / `render_dot` turn that dict into a string. `Environment`
exposes these as `env.to_graph()` and `env.describe(...)`.

This module imports nothing from the package (it duck-types the environment),
so it can be imported from anywhere without cycles.
"""
from typing import Any, Dict, List


def _role_of(component_name: str, port_name: str) -> str:
    """Port role = port name without the '<component name>_' prefix."""
    prefix = f'{component_name}_'
    return port_name[len(prefix):] if port_name.startswith(prefix) else port_name


def build_graph(env) -> Dict[str, Any]:
    """Build a plain-dict graph of the environment's topology."""
    port_owner: Dict[str, str] = {}
    for comp in env.components.values():
        for port_name in comp.ports:
            port_owner[port_name] = comp.name

    category_of = {}
    for category, members in env.components_classified.items():
        for comp in members:
            category_of[comp.name] = category

    components = []
    for comp in env.components.values():
        ports = []
        for port_name, port in comp.ports.items():
            ports.append({
                'name': port_name,
                'role': _role_of(comp.name, port_name),
                'type': type(port).__name__,
                'layers': list(port.layers),
                'connected_to': port.connected_port.name if port.connected_port is not None else None,
            })
        entry = {
            'name': comp.name,
            'type': type(comp).__name__,
            'category': category_of.get(comp.name, 'unknown'),
            'composite': comp.name in env.composite_members,
            'ports': ports,
        }
        members = env.composite_members.get(comp.name)
        if members:
            entry['members'] = list(members)
            exposed = {}
            for m in members:
                internal = env.components.get(m)
                if internal is None:
                    continue
                for port_name in internal.ports:
                    exposed.setdefault(_role_of(comp.name, port_name), port_name)
            entry['exposed_ports'] = exposed
        components.append(entry)

    connections = []
    for port_a, port_b in env.connections:
        port = env.ports.get(port_a)
        connections.append({
            'a': port_a,
            'b': port_b,
            'layers': list(port.layers) if port is not None else [],
            'internal': frozenset((port_a, port_b)) in env.internal_connections,
        })

    sensors = []
    for sensor in env.sensors.values():
        measures = []
        port_name = getattr(sensor, 'port_name', None)
        if port_name is None and getattr(sensor, 'port_spec', None) is not None:
            spec = sensor.port_spec
            port_name = spec if isinstance(spec, str) else getattr(spec, 'name', repr(spec))
        if port_name:
            measures.append(f"port '{port_name}'")
        component_name = getattr(sensor, 'component_name', None)
        if component_name:
            measures.append(f"component '{component_name}'")
        sensors.append({'name': sensor.name, 'type': type(sensor).__name__,
                        'measures': ', '.join(measures) if measures else '?'})

    controllers = []
    for controller in env.controllers.values():
        controllers.append({
            'name': controller.name,
            'type': type(controller).__name__,
            'controls': list(getattr(controller, 'controlled_component_names', [])),
            'reads': list(getattr(controller, 'sensor_names', {}).values()),
        })

    unconnected_ports = [name for name, port in env.ports.items() if port.connected_port is None]

    issues = [f"port '{name}' is not connected (carries zero flow)" for name in unconnected_ports]
    for comp in env.components.values():
        if comp.ports and all(p.connected_port is None for p in comp.ports.values()):
            issues.append(f"component '{comp.name}' has no connected ports")

    return {
        'components': components,
        'port_owner': port_owner,
        'connections': connections,
        'sensors': sensors,
        'controllers': controllers,
        'unconnected_ports': unconnected_ports,
        'issues': issues,
    }


def _member_map(graph) -> Dict[str, str]:
    """internal component name -> composite name"""
    result = {}
    for c in graph['components']:
        for m in c.get('members', []):
            result[m] = c['name']
    return result


def _component(graph, name):
    return next(c for c in graph['components'] if c['name'] == name)


def _edge_node(member_of, owner, internal):
    """External connections of internal members are drawn from the composite node."""
    if not internal and owner in member_of:
        return member_of[owner]
    return owner


def _port_line(owner: str, port: dict, graph) -> str:
    target = port['connected_to']
    if target is None:
        dest = '(unconnected)'
    else:
        target_owner = graph['port_owner'].get(target, '?')
        dest = f'{target_owner}.{_role_of(target_owner, target)}'
    return f"  {port['role']}  {port['type']}({', '.join(port['layers'])}) --> {dest}"


def render_text(graph: Dict[str, Any], expand_composites: bool = True) -> str:
    comps = graph['components']
    n_composite = sum(1 for c in comps if c['composite'])
    n_ports = sum(len(c['ports']) for c in comps)
    n_internal = sum(1 for c in graph['connections'] if c['internal'])
    member_of = _member_map(graph)

    lines = [f"Environment: {len(comps)} components ({n_composite} composite), "
             f"{n_ports} ports, {len(graph['connections'])} connections ({n_internal} internal)"]
    if graph['unconnected_ports']:
        lines.append(f"Unconnected ports (zero flow): {', '.join(graph['unconnected_ports'])}")
    lines.append('')
    lines.append('Components:')
    for c in comps:
        if expand_composites and c['name'] in member_of:
            continue  # shown under its composite
        suffix = f" (internal to {member_of[c['name']]})" if c['name'] in member_of else ''
        lines.append(f"  {c['name']}  [{c['type']} / {c['category']}]{suffix}")
        if expand_composites and c.get('members'):
            members_desc = ', '.join(f"{m} ({_component(graph, m)['type']})" for m in c['members'])
            lines.append(f'    members: {members_desc}')
            exposed = c.get('exposed_ports', {})
            if exposed:
                exposed_desc = ', '.join(f"{r} -> {graph['port_owner'].get(p, '?')}.{p}"
                                         for r, p in exposed.items())
                lines.append(f'    exposed ports: {exposed_desc}')
            role_of_port = {p: r for r, p in exposed.items()}
            for m in c['members']:
                for p in _component(graph, m)['ports']:
                    p = dict(p, role=role_of_port.get(p['name'], p['role']))
                    lines.append('  ' + _port_line(m, p, graph))
        for p in c['ports']:
            lines.append(_port_line(c['name'], p, graph))
    lines.append('')
    lines.append('Connections:')
    for conn in graph['connections']:
        a, b = conn['a'], conn['b']
        oa = graph['port_owner'].get(a, '?')
        ob = graph['port_owner'].get(b, '?')
        tag = ' (internal)' if conn['internal'] else ''
        lines.append(f"  {oa}.{_role_of(oa, a)} -- {ob}.{_role_of(ob, b)}  "
                     f"[{', '.join(conn['layers'])}]{tag}")
    lines.append('')
    lines.append('Sensors:')
    for s in graph['sensors']:
        lines.append(f"  {s['name']} [{s['type']}] -> {s['measures']}")
    lines.append('')
    lines.append('Controllers:')
    for c in graph['controllers']:
        lines.append(f"  {c['name']} [{c['type']}] -> controls {c['controls']}, reads {c['reads']}")
    if graph['issues']:
        lines.append('')
        lines.append('Issues:')
        for issue in graph['issues']:
            lines.append(f'  - {issue}')
    return '\n'.join(lines)


def _mermaid_id(name: str) -> str:
    return ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in name)


def render_mermaid(graph: Dict[str, Any], expand_composites: bool = True) -> str:
    comps = graph['components']
    member_of = _member_map(graph) if expand_composites else {}
    lines = ['graph LR']
    for c in comps:
        if expand_composites and c['name'] in member_of:
            continue
        nid = _mermaid_id(c['name'])
        label = f"{c['name']} ({c['type']})"
        if expand_composites and c.get('members'):
            lines.append(f'  subgraph {nid}["{label}"]')
            for m in c['members']:
                mc = _component(graph, m)
                lines.append(f'    {_mermaid_id(m)}["{m} ({mc["type"]})"]')
            lines.append('  end')
        else:
            lines.append(f'  {nid}["{label}"]')
    for conn in graph['connections']:
        a, b = conn['a'], conn['b']
        oa = graph['port_owner'].get(a, '?')
        ob = graph['port_owner'].get(b, '?')
        na = _edge_node(member_of, oa, conn['internal'])
        nb = _edge_node(member_of, ob, conn['internal'])
        label = _role_of(na, a)
        style = ':::internal' if conn['internal'] else ''
        lines.append(f'  {_mermaid_id(na)} -- "{label}" --> {_mermaid_id(nb)}{style}')
    if any(conn['internal'] for conn in graph['connections']):
        lines.append('  classDef internal stroke-dasharray: 5 5')
    return '\n'.join(lines)


def render_dot(graph: Dict[str, Any], expand_composites: bool = True) -> str:
    comps = graph['components']
    member_of = _member_map(graph) if expand_composites else {}
    lines = ['digraph environment {']
    lines.append('  rankdir=LR;')
    for c in comps:
        if expand_composites and c['name'] in member_of:
            continue
        label = f"{c['name']}\\n({c['type']})"
        if expand_composites and c.get('members'):
            lines.append(f'  subgraph cluster_{_mermaid_id(c["name"])} {{')
            lines.append(f'    label="{label}";')
            for m in c['members']:
                mc = _component(graph, m)
                lines.append(f'    {_mermaid_id(m)} [label="{m}\\n({mc["type"]})"];')
            lines.append('  }')
        else:
            lines.append(f'  {_mermaid_id(c["name"])} [label="{label}"];')
    for conn in graph['connections']:
        a, b = conn['a'], conn['b']
        oa = graph['port_owner'].get(a, '?')
        ob = graph['port_owner'].get(b, '?')
        na = _edge_node(member_of, oa, conn['internal'])
        nb = _edge_node(member_of, ob, conn['internal'])
        style = ' [style=dashed]' if conn['internal'] else ''
        lines.append(f'  {_mermaid_id(na)} -> {_mermaid_id(nb)} [label="{_role_of(na, a)}"]{style};')
    lines.append('}')
    return '\n'.join(lines)


_RENDERERS = {'text': render_text, 'mermaid': render_mermaid, 'dot': render_dot}


def render(graph: Dict[str, Any], format: str = 'text', expand_composites: bool = True) -> str:
    try:
        renderer = _RENDERERS[format]
    except KeyError:
        raise ValueError(f"Unknown format '{format}'. Available: {sorted(_RENDERERS)}")
    return renderer(graph, expand_composites=expand_composites)

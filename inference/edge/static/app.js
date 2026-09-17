/* Local SVG/DOM editor for the device's original Workflow engine. No build step. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const clone = value => JSON.parse(JSON.stringify(value));
  const safeName = /^[A-Za-z][A-Za-z0-9_]{0,63}$/;
  const definitions = {
    input: {title: '图像输入', icon: '▧', category: 'input', type: 'WorkflowImage', output: [['image', '图像', 'image']]},
    crop: {title: '固定区域裁剪', icon: '⌗', category: 'crop', type: 'AbsoluteStaticCrop', aliases: ['roboflow_core/absolute_static_crop@v1'], inputs: [['images', '图像', 'image']], output: [['crops', '裁剪图像', 'image']], defaults: {x_center: 640, y_center: 360, width: 640, height: 480}},
    relative: {title: '比例区域裁剪', icon: '◫', category: 'crop', type: 'RelativeStaticCrop', aliases: ['roboflow_core/relative_statoic_crop@v1'], inputs: [['images', '图像', 'image']], output: [['crops', '裁剪图像', 'image']], defaults: {x_center: 0.5, y_center: 0.5, width: 0.8, height: 0.8}},
    detect: {title: 'RKNN 目标检测', icon: '◉', category: 'model', type: 'ObjectDetectionModel', aliases: ['RoboflowObjectDetectionModel', 'roboflow_core/roboflow_object_detection_model@v1', 'rv1126b/rknn_object_detection@v1'], inputs: [['images', '图像', 'image']], output: [['predictions', '检测结果', 'detections']], defaults: {model_id: '', confidence: 0.25, iou_threshold: 0.45, max_detections: 32, class_agnostic_nms: false}},
    filter: {title: '检测条件过滤', icon: '▽', category: 'filter', type: 'DetectionsFilter', aliases: ['roboflow_core/detections_filter@v1'], inputs: [['predictions', '检测结果', 'detections']], output: [['predictions', '过滤结果', 'detections']], defaults: {operations: [], operations_parameters: {}}},
    classfilter: {title: '分类置信度', icon: '≳', category: 'filter', type: 'roboflow_core/per_class_confidence_filter@v1', inputs: [['predictions', '检测结果', 'detections']], output: [['predictions', '过滤结果', 'detections']], defaults: {class_thresholds: {person: 0.5}, default_threshold: 0.3}},
    dynamic: {title: '按检测框裁剪', icon: '⊡', category: 'crop', type: 'DynamicCrop', aliases: ['Crop', 'roboflow_core/dynamic_crop@v1'], inputs: [['images', '原图', 'image'], ['predictions', '检测框', 'detections']], output: [['crops', '目标图像', 'image']]},
    count: {title: '计数 / 属性提取', icon: '#', category: 'format', type: 'PropertyDefinition', aliases: ['PropertyExtraction', 'roboflow_core/property_definition@v1'], inputs: [['data', '输入数据', 'any']], output: [['output', '数值 / 属性', 'any']], defaults: {operations: [{type: 'SequenceLength'}]}},
    transform: {title: '检测结果变换', icon: '⇄', category: 'filter', type: 'DetectionsTransformation', aliases: ['roboflow_core/detections_transformation@v1'], inputs: [['predictions', '检测结果', 'detections']], output: [['predictions', '变换结果', 'detections']], defaults: {operations: [], operations_parameters: {}}},
    expression: {title: '条件表达式', icon: 'ƒ', category: 'format', type: 'Expression', aliases: ['roboflow_core/expression@v1'], inputs: [['data.value', '输入值', 'any']], output: [['output', '表达式结果', 'any']], defaults: {data: {}, switch: {type: 'CasesDefinition', cases: [], default: {type: 'DynamicCaseResult', parameter_name: 'value'}}}},
    condition: {title: '条件继续', icon: '⋔', category: 'filter', type: 'ContinueIf', aliases: ['roboflow_core/continue_if@v1'], inputs: [['evaluation_parameters.value', '判断值', 'any']], output: [['next', '继续执行', 'control']], defaults: {evaluation_parameters: {}, condition_statement: {type: 'StatementGroup', statements: [{type: 'BinaryStatement', left_operand: {type: 'DynamicOperand', operand_name: 'value'}, comparator: {type: '(Number) >'}, right_operand: {type: 'StaticOperand', value: 0}}]}, next_steps: []}},
    output: {title: '流程输出', icon: '↗', category: 'output', type: 'JsonField', inputs: [['selector', '输出数据', 'any']], output: []},
  };
  const state = {spec: null, id: null, config: {}, positions: {}, selected: null, dirty: false,
    connecting: null, models: [], workflows: [], pipeline: null, pollTimer: null, polling: false, busy: false, epoch: 0};
  function element(tag, className, text) {const e = document.createElement(tag); if (className) e.className = className; if (text !== undefined) e.textContent = text; return e;}
  function getPath(obj, path) {return path.split('.').reduce((value, key) => value && Object.hasOwn(value, key) ? value[key] : undefined, obj);}
  function setPath(obj, path, value) {const parts = path.split('.'); if (parts.some(p => ['__proto__', 'constructor', 'prototype'].includes(p))) throw new Error('无效的参数名称'); let target = obj; for (const part of parts.slice(0, -1)) {if (!Object.hasOwn(target, part) || !target[part] || typeof target[part] !== 'object') target[part] = {}; target = target[part];} target[parts.at(-1)] = value;}
  function descriptor(node, group) {
    if (group === 'inputs') return ['WorkflowImage', 'InferenceImage'].includes(node.type) ? {key: 'input', ...definitions.input} : {key: 'parameter', title: '参数输入', icon: '{}', category: 'input', output: [['value', '参数值', 'any']]};
    if (group === 'outputs') return {key: 'output', ...definitions.output};
    for (const [key, value] of Object.entries(definitions)) if (value.type === node.type || value.aliases?.includes(node.type)) return {key, ...value};
    return {key: 'unknown', title: node.type || '未知节点', icon: '?', category: 'format', inputs: [], output: [['output', '输出', 'any']]};
  }
  function entries() {return ['inputs', 'steps', 'outputs'].flatMap(group => (state.spec?.[group] || []).map(node => ({node, group, id: `${group}:${node.name}`, def: descriptor(node, group)})));}
  function entry(id) {return entries().find(e => e.id === id);}
  function sourceRef(e, output) {return e.group === 'inputs' ? `$inputs.${e.node.name}` : output === 'next' ? `$steps.${e.node.name}` : `$steps.${e.node.name}.${output}`;}
  function ports(e) {
    const fields = (e.def.inputs || []).map(field => [...field]);
    // Imported expressions may contain arbitrary named variables. Keep their
    // references visible and editable rather than discarding those edges.
    for (const key of ['data', 'evaluation_parameters', 'operations_parameters']) {
      const value = e.node[key];
      if (value && typeof value === 'object' && !Array.isArray(value)) for (const name of Object.keys(value)) {
        const path = `${key}.${name}`;
        if (!fields.some(f => f[0] === path)) fields.push([path, name, 'any']);
      }
    }
    if (e.group === 'steps') fields.push(['$control', '条件入口', 'control']);
    return fields;
  }
  function markDirty() {state.dirty = true; $('save-state').textContent = '有未保存的修改';}
  function freshId(prefix = 'workflow') {return prefix + '_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 6);}
  function uniqueName(prefix) {const names = new Set(entries().map(e => e.node.name)); let name = prefix, n = 2; while (names.has(name)) name = prefix + '_' + n++; return name;}
  function defaultModel() {return state.models.find(model => !model.error)?.model_id || '';}
  function filterOperations(confidence = 0.4, classes = []) {
    const statements = [{type: 'BinaryStatement', left_operand: {type: 'DynamicOperand', operations: [{type: 'ExtractDetectionProperty', property_name: 'confidence'}]}, comparator: {type: '(Number) >='}, right_operand: {type: 'StaticOperand', value: confidence}}];
    if (classes.length) statements.push({type: 'BinaryStatement', left_operand: {type: 'DynamicOperand', operations: [{type: 'ExtractDetectionProperty', property_name: 'class_name'}]}, comparator: {type: 'in (Sequence)'}, right_operand: {type: 'StaticOperand', value: classes}});
    return [{type: 'DetectionsFilter', filter_operation: {type: 'StatementGroup', operator: 'and', statements}}];
  }
  function template() {
    return {version: '1.0', inputs: [{type: 'WorkflowImage', name: 'image'}], steps: [
      {type: 'ObjectDetectionModel', name: 'detect', images: '$inputs.image', ...clone(definitions.detect.defaults), model_id: defaultModel()},
      {type: 'PropertyDefinition', name: 'count', data: '$steps.detect.predictions', operations: [{type: 'SequenceLength'}]},
    ], outputs: [{type: 'JsonField', name: 'predictions', selector: '$steps.detect.predictions'}, {type: 'JsonField', name: 'count', selector: '$steps.count.output'}]};
  }
  function arrange() {
    const all = entries(), depths = new Map();
    for (let pass = 0; pass < all.length; pass++) for (const e of all) {
      const refs = ports(e).map(p => getPath(e.node, p[0])).filter(v => typeof v === 'string');
      const parents = all.filter(p => p.id !== e.id && (p.def.output || []).some(o => refs.includes(sourceRef(p, o[0]))));
      const depth = e.group === 'inputs' ? 0 : Math.min(10, parents.length ? Math.max(...parents.map(p => depths.get(p.id) || 0)) + 1 : 1);
      depths.set(e.id, depth);
    }
    const rows = {};
    for (const e of all) {const depth = depths.get(e.id) || 0, row = rows[depth] || 0; state.positions[e.id] = {x: 45 + depth * 265, y: 80 + row * 220}; rows[depth] = row + 1;}
    renderGraph();
  }
  function resetEditor(spec, config = {}, id = null) {
    if (!spec || !Array.isArray(spec.inputs) || !Array.isArray(spec.steps) || !Array.isArray(spec.outputs)) throw new Error('文件需要有效的 Workflow inputs、steps 和 outputs 数组');
    if (spec.steps.length > 32 || spec.inputs.length > 32 || spec.outputs.length > 32) throw new Error('流程节点超过本地画布限制');
    const all = [...spec.inputs, ...spec.steps, ...spec.outputs];
    if (all.some(n => !n || typeof n !== 'object' || !safeName.test(n.name))) throw new Error('节点名称需以英文字母开头，仅含字母、数字、下划线，最多 64 字符');
    for (const group of [spec.inputs, spec.steps, spec.outputs]) if (new Set(group.map(n => n.name)).size !== group.length) throw new Error('同一组中存在重复节点名称');
    state.spec = clone(spec); state.config = clone(config); state.id = id; state.positions = clone(config.edge_ui?.positions || {}); state.connecting = null; state.dirty = false;
    for (const node of state.spec.steps) if (descriptor(node, 'steps').inputs?.some(p => p[0] === 'images') && node.images === undefined && node.image !== undefined) {node.images = node.image; delete node.image;}
    state.selected = spec.steps[0] ? `steps:${spec.steps[0].name}` : `inputs:${spec.inputs[0]?.name}`;
    $('workflow-name').value = config.name || '目标检测与计数'; $('save-state').textContent = id ? '已保存到设备' : '尚未保存';
    if (!Object.keys(state.positions).length) arrange(); else renderGraph();
    renderInspector(); renderList();
  }
  function position(e, index) {
    const p = state.positions[e.id];
    if (p && Number.isFinite(p.x) && Number.isFinite(p.y)) return {x: Math.max(15, Math.min(2140, p.x)), y: Math.max(20, Math.min(960, p.y))};
    return {x: 45 + (index % 4) * 260, y: 70 + Math.floor(index / 4) * 220};
  }
  function freePosition(proposed, ownId) {
    let p = {x: Math.max(15, Math.min(2100, proposed.x)), y: Math.max(20, Math.min(950, proposed.y))};
    const others = Object.entries(state.positions).filter(([id]) => id !== ownId).map(([, value]) => value);
    for (let attempt = 0; attempt < 50; attempt++) {
      if (!others.some(other => Math.abs(other.x - p.x) < 235 && Math.abs(other.y - p.y) < 200)) return p;
      p.y += 220; if (p.y > 950) {p.y = 70; p.x = p.x + 265 > 2100 ? 45 : p.x + 265;}
    }
    return p;
  }
  function renderGraph() {
    $('nodes').replaceChildren();
    entries().forEach((e, index) => {
      const p = position(e, index); state.positions[e.id] = p;
      const card = element('div', `node ${e.def.category}${state.selected === e.id ? ' selected' : ''}`); card.dataset.nodeId = e.id; card.style.left = p.x + 'px'; card.style.top = p.y + 'px';
      const heading = element('div', 'node-heading'); heading.append(element('span', 'node-icon', e.def.icon)); const titles = element('div'); titles.append(element('strong', '', e.def.title), element('small', '', e.node.name)); heading.append(titles); card.append(heading);
      const detail = e.def.key === 'detect' ? (e.node.model_id || '请选择 RKNN 模型') : e.def.key === 'count' ? '输出检测数量或提取属性' : e.def.key === 'input' ? '设备摄像头 · 图像输入' : e.def.key === 'parameter' ? '默认值与类型见高级参数' : e.group === 'outputs' ? (e.node.selector || '连接要返回的数据') : (e.def.key === 'unknown' ? '通过高级参数编辑' : '点击节点设置参数');
      card.append(element('div', 'node-detail', detail));
      const container = element('div', 'node-ports');
      for (const [field, title, kind] of ports(e)) {
        const row = element('div', 'port-row'); const button = element('button', 'port in'); button.type = 'button'; button.dataset.port = `${e.id}|in|${field}`; button.title = `连接到 ${e.node.name} · ${title}`; button.setAttribute('aria-label', button.title);
        button.addEventListener('click', event => {event.stopPropagation(); connectTo(e, field, kind);}); row.append(button, element('span', '', title)); container.append(row);
      }
      for (const [field, title, kind] of e.def.output || []) {
        const row = element('div', 'port-row output-port'); const button = element('button', 'port out'); button.type = 'button'; button.dataset.port = `${e.id}|out|${field}`; button.title = `从 ${e.node.name} · ${title} 连线`; button.setAttribute('aria-label', button.title);
        if (state.connecting?.id === e.id && state.connecting.field === field) button.classList.add('connecting');
        button.addEventListener('click', event => {event.stopPropagation(); state.connecting = {id: e.id, field, kind}; $('connect-hint').hidden = false; renderGraph();}); row.append(element('span', '', title), button); container.append(row);
      }
      card.append(container); card.addEventListener('click', () => selectNode(e.id)); heading.addEventListener('pointerdown', event => beginDrag(event, e.id, card)); $('nodes').append(card);
    });
    $('connect-hint').hidden = !state.connecting;
    renderEdges();
  }
  function selectNode(id) {state.selected = id; for (const node of $('nodes').children) node.classList.toggle('selected', node.dataset.nodeId === id); renderInspector();}
  function beginDrag(event, id, card) {
    if (event.button !== 0) return;
    event.preventDefault(); selectNode(id);
    const original = {...state.positions[id]}; let moved = false;
    const move = e => {const x = Math.max(15, Math.min(2140, original.x + e.clientX - event.clientX)), y = Math.max(20, Math.min(960, original.y + e.clientY - event.clientY)); state.positions[id] = {x, y}; card.style.left = x + 'px'; card.style.top = y + 'px'; moved = true; renderEdges();};
    const stop = () => {document.removeEventListener('pointermove', move); document.removeEventListener('pointerup', stop); document.removeEventListener('pointercancel', stop); if (moved) markDirty();};
    document.addEventListener('pointermove', move); document.addEventListener('pointerup', stop); document.addEventListener('pointercancel', stop);
  }
  function renderEdges() {
    const svg = $('edges'); svg.replaceChildren(); const base = $('canvas').getBoundingClientRect(); const portMap = new Map([...document.querySelectorAll('[data-port]')].map(p => [p.dataset.port, p]));
    function line(from, to) {
      const a = portMap.get(from), b = portMap.get(to); if (!a || !b) return;
      const ar = a.getBoundingClientRect(), br = b.getBoundingClientRect(); const x1 = ar.left + ar.width / 2 - base.left, y1 = ar.top + ar.height / 2 - base.top, x2 = br.left + br.width / 2 - base.left, y2 = br.top + br.height / 2 - base.top; const bend = Math.max(50, Math.abs(x2 - x1) / 2);
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path'); path.setAttribute('d', `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`); svg.append(path);
    }
    const all = entries();
    for (const e of all) for (const [field] of ports(e)) {
      if (field === '$control') continue;
      const value = getPath(e.node, field);
      for (const source of all) for (const [output] of source.def.output || []) if (value === sourceRef(source, output)) line(`${source.id}|out|${output}`, `${e.id}|in|${field}`);
    }
    for (const source of all.filter(e => e.def.key === 'condition')) for (const target of source.node.next_steps || []) {
      const e = all.find(x => x.group === 'steps' && `$steps.${x.node.name}` === target); if (e) line(`${source.id}|out|next`, `${e.id}|in|$control`);
    }
  }
  function connectTo(target, field, kind) {
    const connection = state.connecting; if (!connection) {selectNode(target.id); EdgeUI.toast('先点击上游节点右侧的输出圆点'); return;}
    const source = entry(connection.id); if (!source) return;
    if (source.id === target.id) return EdgeUI.toast('不能连接节点自身', true);
    if ((kind === 'control') !== (connection.kind === 'control') || (kind !== 'any' && connection.kind !== 'any' && kind !== connection.kind)) return EdgeUI.toast('输入与输出的数据类型不匹配', true);
    if (kind === 'control') {source.node.next_steps = [...new Set([...(source.node.next_steps || []), `$steps.${target.node.name}`])];}
    else setPath(target.node, field, sourceRef(source, connection.field));
    state.connecting = null; markDirty(); renderGraph(); selectNode(target.id);
  }
  function addNode(key) {
    const def = definitions[key], group = key === 'input' ? 'inputs' : key === 'output' ? 'outputs' : 'steps';
    if (state.spec[group].length >= 32) return EdgeUI.toast('该组最多添加 32 个节点', true);
    const name = uniqueName(key === 'input' ? 'image' : key === 'output' ? 'result' : key);
    const node = {type: def.type, name, ...clone(def.defaults || {})};
    for (const [field] of def.inputs || []) setPath(node, field, '');
    if (key === 'detect') node.model_id = defaultModel();
    if (key === 'filter') node.operations = filterOperations();
    const parent = entry(state.selected); const mainInput = def.inputs?.[0], parentOutput = parent?.def.output?.find(p => p[2] !== 'control' && (mainInput?.[2] === 'any' || p[2] === 'any' || p[2] === mainInput?.[2]));
    if (mainInput && parentOutput) setPath(node, mainInput[0], sourceRef(parent, parentOutput[0]));
    state.spec[group].push(node); const id = `${group}:${name}`; const p = parent ? state.positions[parent.id] : null; state.positions[id] = freePosition(p ? {x: p.x + 265, y: p.y + 20} : {x: 70, y: 90 + entries().length * 20}, id); state.selected = id; markDirty(); renderGraph(); renderInspector();
    const scroll = $('canvas-scroll'); scroll.scrollTo({left: Math.max(0, state.positions[id].x - 80), top: Math.max(0, state.positions[id].y - 80), behavior: 'smooth'});
  }
  function field(labelText, value, callback, options = {}) {
    const label = element('label', '', labelText); let input;
    if (options.choices) {input = element('select'); for (const [v, title] of options.choices) {const option = element('option', '', title); option.value = v; input.append(option);} input.value = value ?? '';}
    else {input = element('input'); input.type = options.type || 'text'; input.value = value ?? ''; for (const key of ['min', 'max', 'step', 'placeholder']) if (options[key] !== undefined) input[key] = options[key];}
    input.addEventListener('change', () => {
      try {
        let v = input.value;
        if (options.type === 'number') {v = Number(v); if (!input.value || !Number.isFinite(v) || !input.checkValidity()) throw new Error(labelText + '数值无效');}
        callback(v); markDirty(); renderGraph();
      } catch (error) {EdgeUI.toast(error.message, true); renderInspector();}
    }); label.append(input); return label;
  }
  function replaceSelectors(value, from, to) {
    if (typeof value === 'string') return value === from || value.startsWith(from + '.') ? to + value.slice(from.length) : value;
    if (Array.isArray(value)) return value.map(v => replaceSelectors(v, from, to));
    if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, replaceSelectors(v, from, to)]));
    return value;
  }
  function renameNode(e, name) {
    if (!safeName.test(name)) throw new Error('节点名称需以英文字母开头，仅含字母、数字与下划线');
    if (state.spec[e.group].some(n => n !== e.node && n.name === name)) throw new Error('节点名称已存在');
    const old = e.node.name; e.node.name = name;
    if (e.group !== 'outputs') state.spec = replaceSelectors(state.spec, `${e.group === 'inputs' ? '$inputs' : '$steps'}.${old}`, `${e.group === 'inputs' ? '$inputs' : '$steps'}.${name}`);
    const id = `${e.group}:${name}`; state.positions[id] = state.positions[e.id]; if (id !== e.id) delete state.positions[e.id]; state.selected = id; renderInspector();
  }
  function renderInspector() {
    const pane = $('inspector'), e = entry(state.selected); pane.replaceChildren();
    if (!e) {pane.append(element('p', 'muted small', '选择节点以编辑参数。')); return;}
    const header = element('div', 'inspector-heading'); header.append(element('h3', '', e.def.title)); const remove = element('button', 'icon-button danger', '×'); remove.title = '删除节点'; remove.addEventListener('click', () => {
      if (!confirm(`删除节点「${e.node.name}」？与它相连的输入需要重新连接。`)) return;
      state.spec[e.group] = state.spec[e.group].filter(n => n.name !== e.node.name);
      for (const other of entries()) {
        const prefix = e.group === 'inputs' ? `$inputs.${e.node.name}` : e.group === 'steps' ? `$steps.${e.node.name}` : null;
        for (const [path] of ports(other)) {const value = getPath(other.node, path); if (prefix && typeof value === 'string' && (value === prefix || value.startsWith(prefix + '.'))) setPath(other.node, path, '');}
        if (e.group === 'steps' && Array.isArray(other.node.next_steps)) other.node.next_steps = other.node.next_steps.filter(v => v !== `$steps.${e.node.name}`);
      }
      delete state.positions[e.id]; state.selected = null; markDirty(); renderGraph(); renderInspector();
    }); header.append(remove); pane.append(header);
    pane.append(field('节点名称', e.node.name, v => renameNode(e, v)));
    for (const [path, title, kind] of ports(e)) {
      if (path === '$control') continue;
      const choices = [['', '请选择输入来源']];
      for (const source of entries().filter(x => x.id !== e.id)) for (const [out, outTitle, sourceKind] of source.def.output || []) if (sourceKind !== 'control' && (kind === 'any' || sourceKind === 'any' || sourceKind === kind)) choices.push([sourceRef(source, out), `${source.node.name} · ${outTitle}`]);
      const value = getPath(e.node, path) || ''; if (value && !choices.some(c => c[0] === value)) choices.push([value, value + '（检查来源）']);
      pane.append(field(title + '来源', value, v => setPath(e.node, path, v), {choices}));
    }
    const numeric = (name, title, min, max, step = 'any') => pane.append(field(title, e.node[name], v => {e.node[name] = v;}, {type: 'number', min, max, step}));
    if (e.def.key === 'detect') {
      const choices = [['', '请选择设备 RKNN 模型'], ...state.models.filter(m => !m.error).map(m => [m.model_id, m.model_id])];
      if (e.node.model_id && !choices.some(c => c[0] === e.node.model_id)) choices.push([e.node.model_id, e.node.model_id + '（未安装）']);
      pane.append(field('RKNN 模型', e.node.model_id, v => {e.node.model_id = v;}, {choices})); numeric('confidence', '置信度阈值', 0, 1, 0.01); numeric('iou_threshold', 'NMS 重叠阈值', 0, 1, 0.01); numeric('max_detections', '最大目标数', 1, 1000, 1);
      pane.append(field('保留类别（可选）', Array.isArray(e.node.class_filter) ? e.node.class_filter.join(', ') : e.node.class_filter || '', v => {e.node.class_filter = v.trim().startsWith('$') ? v.trim() : v.trim() ? v.split(/[,，]/).map(x => x.trim()).filter(Boolean) : null;}, {placeholder: 'person, car'}));
    }
    if (['crop', 'relative'].includes(e.def.key)) {const relative = e.def.key === 'relative'; for (const [name, title] of [['x_center', '中心 X'], ['y_center', '中心 Y'], ['width', '宽度'], ['height', '高度']]) numeric(name, title + (relative ? '（比例 0–1）' : '（像素）'), relative ? 0 : 1, relative ? 1 : 16000, relative ? 0.01 : 1);}
    if (e.def.key === 'filter') {
      const statements = e.node.operations?.[0]?.filter_operation?.statements || [];
      const score = statements.find(s => s.left_operand?.operations?.[0]?.property_name === 'confidence'); const category = statements.find(s => s.left_operand?.operations?.[0]?.property_name === 'class_name');
      let confidence = typeof score?.right_operand?.value === 'number' ? score.right_operand.value : 0.4; let classes = Array.isArray(category?.right_operand?.value) ? category.right_operand.value : [];
      pane.append(field('最低置信度', confidence, v => {confidence = v; e.node.operations = filterOperations(confidence, classes);}, {type: 'number', min: 0, max: 1, step: 0.01}));
      pane.append(field('保留类别（空值表示全部）', classes.join(', '), v => {classes = v.split(/[,，]/).map(x => x.trim()).filter(Boolean); e.node.operations = filterOperations(confidence, classes);}, {placeholder: 'person, car'}));
      pane.append(element('p', 'help', '修改这两项会更新为“置信度且类别”过滤。复杂条件请使用高级参数。'));
    }
    if (e.def.key === 'classfilter') {numeric('default_threshold', '其他类别最低置信度', 0, 1, 0.01); pane.append(element('p', 'help', '各类别阈值通过高级参数中的 class_thresholds 配置。'));}
    if (e.def.key === 'count') {
      const isCount = e.node.operations?.length === 1 && e.node.operations[0].type === 'SequenceLength';
      pane.append(field('提取内容', isCount ? 'count' : 'custom', v => {if (v === 'count') e.node.operations = [{type: 'SequenceLength'}]; else openAdvanced();}, {choices: [['count', '目标数量'], ['custom', '高级属性提取…']]}));
    }
    if (e.def.key === 'condition') {
      pane.append(element('p', 'help', '默认在输入值大于 0 时继续。把“继续执行”端口连接到下游节点的“条件入口”；高级参数可修改判断。'));
      for (const target of e.node.next_steps || []) {const button = element('button', 'ghost small full', '移除控制连接 ' + target); button.addEventListener('click', () => {e.node.next_steps = e.node.next_steps.filter(t => t !== target); markDirty(); renderGraph(); renderInspector();}); pane.append(button);}
    }
    if (e.group === 'outputs') pane.append(element('p', 'help', '摄像头流程输出检测和数值等 JSON 数据。图像输出会超出设备视频结果契约。'));
    if (e.def.key === 'dynamic') pane.append(element('p', 'help', '每帧最多 32 次裁剪，总像素受设备预算限制。'));
    pane.append(element('hr'));
    const advanced = element('button', 'ghost full small', '{ } 高级节点参数'); advanced.addEventListener('click', openAdvanced); pane.append(advanced);
  }
  function openAdvanced() {const e = entry(state.selected); if (!e) return; $('advanced-json').value = JSON.stringify(e.node, null, 2); $('advanced-error').textContent = ''; $('json-dialog').showModal();}
  async function refreshModels() {const response = await EdgeUI.request('/model/registry'); state.models = response.models || []; $('model-summary').textContent = `${state.models.filter(m => !m.error).length} 个已安装 RKNN 模型 · RV1126B`;}
  async function refreshList() {const response = await EdgeUI.request('/build/api'); state.workflows = Object.entries(response.data || {}).map(([id, row]) => ({id, ...row})); renderList();}
  function renderList() {
    $('workflow-list').replaceChildren();
    for (const row of state.workflows) {const button = element('button', 'workflow-item' + (state.id === row.id ? ' selected' : ''), row.config?.name || row.id); button.title = row.config?.name || row.id; button.addEventListener('click', () => action(async () => {
      if (!discardChanges()) return; const response = await EdgeUI.request('/build/api/' + encodeURIComponent(row.id)); const config = response.data.config;
      const spec = config.specification || config.workflow || config.definition;
      resetEditor(typeof spec === 'string' ? JSON.parse(spec) : spec, config, row.id);
    })); $('workflow-list').append(button);}
    if (!state.workflows.length) $('workflow-list').append(element('p', 'palette-tip muted', '还没有保存的流程。完成编排后点击“保存流程”。'));
  }
  function discardChanges() {return !state.dirty || confirm('当前修改尚未保存，是否放弃这些修改？');}
  async function validate() {await EdgeUI.request('/workflows/validate', state.spec);}
  async function action(operation) {
    if (state.busy) return;
    state.busy = true;
    try {await operation();} catch (error) {EdgeUI.toast(error.message, true);} finally {state.busy = false;}
  }
  async function save() {
    const name = $('workflow-name').value.trim(); if (!name) throw new Error('请输入流程名称');
    await validate(); const id = state.id || freshId();
    const config = {...state.config, id, name, specification: clone(state.spec), edge_ui: {...state.config.edge_ui, positions: clone(state.positions)}};
    await EdgeUI.request('/build/api/' + encodeURIComponent(id), config); state.config = config; state.id = id; state.dirty = false; $('save-state').textContent = '已校验并保存到设备'; await refreshList(); EdgeUI.toast('流程已保存');
  }
  function updatePipeline(pipeline) {
    state.pipeline = pipeline || null;
    const active = pipeline && ['running', 'paused'].includes(pipeline.status); const paused = pipeline?.status === 'paused';
    $('start-video').disabled = !!active; $('pause-video').disabled = !active; $('stop-video').disabled = !pipeline || pipeline.status === 'stopped'; $('pause-video').textContent = paused ? '恢复' : '暂停';
    $('pipeline-state').textContent = !pipeline ? '尚未启动' : ({running: '正在运行 · 编辑将在下次启动时生效', paused: '已暂停', stopped: '已停止 · 模型已释放', failed: '运行失败'}[pipeline.status] || pipeline.status) + (pipeline.error ? '：' + pipeline.error : '');
    $('queue-count').replaceChildren(document.createTextNode(String(pipeline?.retained_results || 0) + ' '), element('small', '', '/ 2'));
  }
  function showResults(records) {
    if (!records.length) return;
    const record = records.at(-1); const result = record.result ?? record;
    $('result-json').textContent = JSON.stringify(records.slice(-2), null, 2).slice(0, 100000);
    const output = Array.isArray(result) ? result[0] : result; let detections = [];
    if (output?.predictions && Array.isArray(output.predictions)) detections = output.predictions;
    else if (output && typeof output === 'object') for (const value of Object.values(output)) if (value && Array.isArray(value.predictions)) {detections = value.predictions; break;}
    $('detection-count').textContent = String(typeof output?.count === 'number' ? output.count : detections.length);
    const container = $('result-summary'); container.replaceChildren();
    if (detections.length) {
      const table = element('table', 'result-table'), heading = element('tr'); for (const label of ['类别', '置信度', '位置 / 尺寸']) heading.append(element('th', '', label)); table.append(heading);
      for (const prediction of detections.slice(0, 12)) {const row = element('tr'); const score = Number(prediction.confidence); row.append(element('td', '', prediction.class || String(prediction.class_id)), element('td', '', Number.isFinite(score) ? (score * 100).toFixed(1) + '%' : '—'), element('td', '', ['x', 'y', 'width', 'height'].map(k => Number(prediction[k] || 0).toFixed(0)).join(' / '))); table.append(row);} container.append(table);
    } else {container.append(element('p', 'muted small', '本帧未输出检测目标。'));}
    if (output && typeof output === 'object') {const values = Object.entries(output).filter(([, value]) => value === null || ['string', 'number', 'boolean'].includes(typeof value)); if (values.length) container.append(element('p', 'palette-tip', values.map(([k, v]) => `${k}: ${v}`).join('  ·  ')));}
  }
  async function poll() {
    if (state.polling || !EdgeUI.session) return;
    if (document.hidden || state.busy) {state.pollTimer = setTimeout(poll, 1000); return;}
    state.polling = true; const epoch = state.epoch;
    try {
      const listing = await EdgeUI.request('/inference_pipelines/list'); if (epoch !== state.epoch) return;
      const pipeline = listing.pipelines?.[0] || null; updatePipeline(pipeline);
      if (pipeline && ['running', 'paused'].includes(pipeline.status)) {const result = await EdgeUI.request('/inference_pipelines/consume', {pipeline_id: pipeline.pipeline_id}); if (epoch === state.epoch) showResults(result.results || []);}
    } catch (error) {if (error.status !== 401) $('pipeline-state').textContent = '状态读取失败：' + error.message;}
    finally {state.polling = false; if (EdgeUI.session) state.pollTimer = setTimeout(poll, 1000);}
  }
  function showLogin() {state.epoch++; clearTimeout(state.pollTimer); $('login-screen').hidden = false; $('workspace').hidden = true;}
  async function openWorkspace(session) {
    $('login-screen').hidden = true; $('workspace').hidden = false; const settings = session.settings || {};
    const fps = settings.max_fps || 10; $('video-fps').max = String(fps); $('video-fps').value = String(Math.min(3, fps)); $('settings-values').replaceChildren();
    for (const [title, value] of [['监听地址', settings.host || location.hostname], ['端口', settings.port || location.port], ['帧率上限', fps + ' FPS'], ['流程节点上限', settings.max_workflow_steps || 32]]) { $('settings-values').append(element('dt', 'muted', title), element('dd', '', String(value))); }
    await refreshModels(); await refreshList();
    if (!state.spec) resetEditor(template()); else {renderGraph(); renderInspector();}
    clearTimeout(state.pollTimer); poll();
  }
  for (const [key, def] of Object.entries(definitions)) {const button = element('button', 'palette-button'); button.append(element('span', '', def.icon), document.createTextNode(def.title)); button.title = '添加' + def.title; button.addEventListener('click', () => addNode(key)); $('node-palette').append(button);}
  $('workflow-name').addEventListener('input', markDirty);
  $('login-form').addEventListener('submit', async event => {event.preventDefault(); const token = $('login-token').value; $('login-token').value = ''; $('login-error').textContent = ''; try {await openWorkspace(await EdgeUI.login(token));} catch (error) {$('login-error').textContent = error.message;}});
  $('logout').addEventListener('click', () => action(async () => {if (!discardChanges()) return; await EdgeUI.logout(); state.spec = null; state.dirty = false; showLogin();}));
  $('settings-toggle').addEventListener('click', () => {$('settings-panel').hidden = !$('settings-panel').hidden;});
  $('new-workflow').addEventListener('click', () => {if (!discardChanges()) return; resetEditor({version: '1.0', inputs: [{type: 'WorkflowImage', name: 'image'}], steps: [], outputs: []}, {name: '新的流程'}); markDirty();});
  $('reset-template').addEventListener('click', () => {if (!discardChanges()) return; resetEditor(template()); markDirty();});
  $('arrange-nodes').addEventListener('click', () => {arrange(); markDirty();});
  $('validate-workflow').addEventListener('click', () => action(async () => {await validate(); EdgeUI.toast('流程校验通过');}));
  $('save-workflow').addEventListener('click', () => action(save));
  $('delete-workflow').addEventListener('click', () => action(async () => {if (!state.id) throw new Error('当前流程尚未保存'); if (!confirm('从设备删除当前保存的流程？')) return; await EdgeUI.request('/build/api/' + encodeURIComponent(state.id), undefined, 'DELETE'); await refreshList(); resetEditor(template()); EdgeUI.toast('流程已删除');}));
  $('export-workflow').addEventListener('click', () => {const body = {...state.config, id: state.id || freshId(), name: $('workflow-name').value, specification: state.spec, edge_ui: {positions: state.positions}}; const url = URL.createObjectURL(new Blob([JSON.stringify(body, null, 2)], {type: 'application/json'})); const link = element('a'); link.href = url; link.download = (state.id || 'workflow') + '.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);});
  $('import-workflow').addEventListener('click', () => $('import-file').click());
  $('import-file').addEventListener('change', () => action(async () => {const file = $('import-file').files[0]; $('import-file').value = ''; if (!file || !discardChanges()) return; if (file.size > 1024 * 1024) throw new Error('请选择小于 1 MiB 的流程 JSON'); const config = JSON.parse(await file.text()); const spec = config.specification || config.workflow || config.definition || config; resetEditor(typeof spec === 'string' ? JSON.parse(spec) : spec, config.specification ? config : {name: file.name.replace(/\.json$/i, '')}); markDirty();}));
  $('apply-advanced').addEventListener('click', () => {
    try {const e = entry(state.selected), value = JSON.parse($('advanced-json').value); if (!value || typeof value !== 'object' || Array.isArray(value) || !safeName.test(value.name) || typeof value.type !== 'string') throw new Error('需要包含有效 type、name 的节点对象'); if (state.spec[e.group].some(n => n !== e.node && n.name === value.name)) throw new Error('节点名称已存在'); const oldName = e.node.name; const index = state.spec[e.group].indexOf(e.node); state.spec[e.group][index] = value; const newId = `${e.group}:${value.name}`; state.positions[newId] = state.positions[e.id]; if (newId !== e.id) {delete state.positions[e.id]; if (e.group !== 'outputs') state.spec = replaceSelectors(state.spec, `${e.group === 'inputs' ? '$inputs' : '$steps'}.${oldName}`, `${e.group === 'inputs' ? '$inputs' : '$steps'}.${value.name}`);} state.selected = newId; markDirty(); renderGraph(); renderInspector(); $('json-dialog').close();} catch (error) {$('advanced-error').textContent = error.message;}
  });
  $('start-video').addEventListener('click', () => action(async () => {
    const inputs = state.spec.inputs.filter(n => ['WorkflowImage', 'InferenceImage'].includes(n.type)); if (inputs.length !== 1) throw new Error('摄像头流程需要且只能有一个图像输入'); const fps = Number($('video-fps').value); if (!Number.isFinite(fps) || fps <= 0 || fps > Number($('video-fps').max)) throw new Error('帧率超出设备上限'); await validate();
    const response = await EdgeUI.request('/inference_pipelines/initialise', {specification: state.spec, image_input: inputs[0].name, max_fps: fps}); updatePipeline({...response, frames: 0, retained_results: 0}); EdgeUI.toast('摄像头流程已启动');
  }));
  $('pause-video').addEventListener('click', () => action(async () => {if (!state.pipeline) return; const command = state.pipeline.status === 'paused' ? 'resume' : 'pause'; updatePipeline(await EdgeUI.request('/inference_pipelines/' + command, {pipeline_id: state.pipeline.pipeline_id}));}));
  $('stop-video').addEventListener('click', () => action(async () => {if (!state.pipeline) return; updatePipeline(await EdgeUI.request('/inference_pipelines/terminate', {pipeline_id: state.pipeline.pipeline_id})); EdgeUI.toast('流程已停止，模型资源已释放');}));
  window.addEventListener('keydown', event => {if (event.key === 'Escape') {state.connecting = null; renderGraph();}});
  window.addEventListener('resize', renderEdges);
  window.addEventListener('edge-session-expired', showLogin);
  window.addEventListener('beforeunload', event => {if (state.dirty) {event.preventDefault(); event.returnValue = '';}});
  EdgeUI.refreshSession().then(openWorkspace).catch(error => {showLogin(); if (error.status !== 401) $('login-error').textContent = error.message;});
})();

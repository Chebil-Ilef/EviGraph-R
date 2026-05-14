const GRAPH_DATA = {
  nodes: __NODES__,
  edges: __EDGES__
};

const TYPE_COLOR = {
  paper:   '#3b82f6',
  chunk:   '#06b6d4',
  claim:   '#f59e0b',
};
const TYPE_COLOR_DIM = {
  paper:   '#1e3a5f',
  chunk:   '#0c3b45',
  claim:   '#4a3000',
};
const EDGE_COLOR = {
  CHUNK_OF:       '#354e70',
  cites:          '#1e40af',
  extracted_from: '#78470f',
  supports:       '#14532d',
};

function _nodeSize(ele) {
  const t = ele.data('type');
  if (t === 'paper')   return 120;
  if (t === 'chunk')   return 100;
  if (t === 'claim')   return 90;
  return 90;
}

function _textMaxWidth(ele) {
  return Math.round(_nodeSize(ele) * 0.82) + 'px';
}

function _fontSize(ele) {
  const label = ele.data('label') ?? '';
  const len   = label.length;
  const base  = ele.data('type') === 'paper' ? 13 : 12;
  if (len > 30) return (base - 3) + 'px';
  if (len > 20) return (base - 2) + 'px';
  if (len > 12) return (base - 1) + 'px';
  return base + 'px';
}

function _nodeShape(ele) {
  const t = ele.data('type');
  if (t === 'paper')   return 'round-rectangle';
  if (t === 'chunk')   return 'round-rectangle';
  if (t === 'claim')   return 'ellipse';
  return 'ellipse';
}

const CY_STYLE = [
  {
    selector: 'node',
    style: {
      'label':              'data(label)',
      'font-size':          ele => _fontSize(ele),
      'font-family':        'Inter, system-ui, sans-serif',
      'text-wrap':          'wrap',
      'text-max-width':     ele => _textMaxWidth(ele),
      'text-valign':        'center',
      'text-halign':        'center',
      'color':              '#e2e8f0',
      'text-outline-width': 0,          // no outline — cleaner inside shape
      'border-width':       2,
      'border-color':       ele => TYPE_COLOR[ele.data('type')] ?? '#64748b',
      'background-color':   ele => TYPE_COLOR_DIM[ele.data('type')] ?? '#1e2333',
      'width':              ele => _nodeSize(ele),
      'height':             ele => _nodeSize(ele),
      'shape':              ele => _nodeShape(ele),
      'padding':            '10px',
      'transition-property': 'background-color, border-color, width, height, opacity',
      'transition-duration': '150ms',
    }
  },
  // per-type tweaks
  { selector: 'node[type="paper"]',   style: { 'shape': 'round-rectangle' } },
  { selector: 'node[type="chunk"]',   style: { 'shape': 'round-rectangle' } },
  { selector: 'node[type="claim"]',   style: { 'shape': 'ellipse' } },

  // verdict border colours (claim nodes after judging) — matches CSS vars
  { selector: 'node[verdict="Supported"]',     style: { 'border-color': '#5fbf82', 'border-width': 3 } },
  { selector: 'node[verdict="Contradicted"]',  style: { 'border-color': '#c7274a', 'border-width': 3 } },
  { selector: 'node[verdict="Not-Supported"]', style: { 'border-color': '#b10505', 'border-width': 3 } },
  { selector: 'node[verdict="Inconclusive"]',  style: { 'border-color': '#94a3b8', 'border-width': 3 } },

  // selected node
  {
    selector: 'node.selected',
    style: {
      'background-color': ele => TYPE_COLOR[ele.data('type')] ?? '#6366f1',
      'border-width': 3,
      'border-color': '#ffffff',
      'z-index': 999,
    }
  },
  // highlighted neighbourhood
  {
    selector: 'node.highlighted',
    style: {
      'background-color': ele => TYPE_COLOR[ele.data('type')] ?? '#64748b',
      'border-color':     ele => TYPE_COLOR[ele.data('type')] ?? '#64748b',
      'opacity': 1,
    }
  },
  // dimmed
  { selector: 'node.dimmed', style: { 'opacity': 0.12 } },

  // default edge
  {
    selector: 'edge',
    style: {
      'width':              1.5,
      'line-color':         ele => EDGE_COLOR[ele.data('relation')] ?? '#2a2f42',
      'target-arrow-color': ele => EDGE_COLOR[ele.data('relation')] ?? '#2a2f42',
      'target-arrow-shape': 'triangle',
      'curve-style':        'bezier',
      'arrow-scale':        0.8,
      'label':              '',
      'font-size':          '11px',
      'color':              '#677383',
      'text-background-color':   '#181c27',
      'text-background-opacity': 1,
      'text-background-padding': '2px',
      'transition-property': 'line-color, opacity',
      'transition-duration': '150ms',
    }
  },
  // selected edge: show label
  {
    selector: 'edge.selected',
    style: {
      'label':              'data(relation)',
      'width':              2.5,
      'line-color':         '#6366f1',
      'target-arrow-color': '#6366f1',
      'z-index':            999,
    }
  },
  { selector: 'edge.highlighted', style: { 'width': 2, 'opacity': 1 } },
  { selector: 'edge.dimmed',      style: { 'opacity': 0.04 } },
];

const cy = cytoscape({
  container: document.getElementById('cy'),
  elements:  GRAPH_DATA,
  style:     CY_STYLE,
  layout:    { name: 'dagre', rankDir: 'LR', nodeSep: 80, rankSep: 140, padding: 60 },
  minZoom:   0.08,
  maxZoom:   4,
  wheelSensitivity: 0.25,
  userZoomingEnabled:  true,
  userPanningEnabled:  true,
  boxSelectionEnabled: false,
});

// state
let selectedId      = null;
let neighborMode    = false;
let breadcrumbs     = [];
let hiddenTypes     = new Set();
let hiddenVerdicts  = new Set();

// Show verdict-specific filters only in after-judging graph
if (IS_JUDGED) {
  document.getElementById('btn-filter-claim').style.display = 'none';
  const vf = document.getElementById('verdict-filters');
  vf.style.display = 'flex';
}

function updateStats() {
  const vis = cy.nodes(':visible').length;
  document.getElementById('stat-nodes').textContent   = `${cy.nodes().length} nodes`;
  document.getElementById('stat-edges').textContent   = `${cy.edges().length} edges`;
  document.getElementById('stat-visible').textContent = `${vis} visible`;
}
updateStats();

function runLayout(name) {
  const opts = {
    dagre:      { name: 'dagre',      rankDir: 'LR', nodeSep: 80, rankSep: 140, padding: 60 },
    cola:       { name: 'cola',       animate: false, nodeSpacing: 60, padding: 60 },
    concentric: { name: 'concentric', concentric: n => _nodeSize(n), levelWidth: () => 2, padding: 60 },
  };
  cy.layout(opts[name] ?? opts.dagre).run();
}

document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    btn.classList.toggle('active');
    const active = btn.classList.contains('active');

    if (btn.dataset.verdict) {
      // verdict-based filter for claim nodes
      const v = btn.dataset.verdict;
      if (active) {
        hiddenVerdicts.delete(v);
      } else {
        hiddenVerdicts.add(v);
      }
      _applyVerdictFilter();
    } else {
      // type-based filter
      const t = btn.dataset.type;
      if (active) {
        hiddenTypes.delete(t);
        cy.nodes(`[type="${t}"]`).removeClass('hidden-type').style('display', 'element');
      } else {
        hiddenTypes.add(t);
        cy.nodes(`[type="${t}"]`).addClass('hidden-type').style('display', 'none');
      }
    }
    updateStats();
  });
});

function _applyVerdictFilter() {
  // Claim nodes without any verdict (no judging yet) are always shown.
  cy.nodes('[type="claim"]').forEach(n => {
    const v = n.data('verdict');
    // un-judged claims (verdict=null/undefined): never hidden by verdict filter
    if (!v) { n.removeClass('hidden-verdict').style('display', 'element'); return; }
    if (hiddenVerdicts.has(v)) {
      n.addClass('hidden-verdict').style('display', 'none');
    } else {
      n.removeClass('hidden-verdict').style('display', 'element');
    }
  });
}

document.getElementById('layout-select').addEventListener('change', e => {
  runLayout(e.target.value);
});

document.getElementById('btn-fit').addEventListener('click', () => {
  cy.fit(cy.nodes(':visible'), 50);
});

document.getElementById('btn-neighborhood').addEventListener('click', () => {
  neighborMode = !neighborMode;
  document.getElementById('btn-neighborhood').classList.toggle('on', neighborMode);
  if (!neighborMode) clearHighlight();
  else if (selectedId) applyNeighborHighlight(selectedId);
});

function applyNeighborHighlight(nodeId) {
  const sel  = cy.getElementById(nodeId);
  const hood = sel.closedNeighborhood();
  cy.elements().addClass('dimmed').removeClass('highlighted selected');
  hood.removeClass('dimmed').addClass('highlighted');
  sel.removeClass('dimmed').removeClass('highlighted').addClass('selected');
}

function clearHighlight() {
  cy.elements().removeClass('dimmed highlighted selected');
  if (selectedId) cy.getElementById(selectedId).addClass('selected');
}

document.getElementById('search-input').addEventListener('input', e => {
  const q = e.target.value.trim().toLowerCase();
  if (!q) { cy.elements().removeClass('dimmed'); return; }
  const matches = cy.nodes().filter(n =>
    n.data('label').toLowerCase().includes(q) ||
    n.data('full_text').toLowerCase().includes(q) ||
    n.data('paper_id').toLowerCase().includes(q)
  );
  if (matches.length === 0) return;
  cy.elements().addClass('dimmed');
  matches.removeClass('dimmed').addClass('highlighted');
  cy.animate({ fit: { eles: matches, padding: 70 } }, { duration: 400 });
});

cy.on('tap', 'node', function(evt) {
  const node = evt.target;
  const id   = node.id();
  if (selectedId === id) return;

  cy.nodes().removeClass('selected');
  node.addClass('selected');
  selectedId = id;

  if (neighborMode) applyNeighborHighlight(id);

  pushBreadcrumb(id, node.data('label'));
  renderInspector(node);
});

cy.on('tap', 'edge', function(evt) {
  const edge = evt.target;
  cy.edges().removeClass('selected');
  edge.addClass('selected');
  renderEdgeInspector(edge);
});

cy.on('tap', function(evt) {
  if (evt.target === cy) {
    cy.elements().removeClass('selected dimmed highlighted');
    selectedId = null;
    if (!neighborMode) clearHighlight();
    showEmptyInspector();
  }
});


const TYPE_LABEL = { paper: 'Paper', chunk: 'Chunk', claim: 'Claim' };

function showEmptyInspector() {
  document.getElementById('inspector-type-badge').style.display = 'none';
  document.getElementById('inspector-node-id').textContent = '—';
  document.getElementById('inspector-title').textContent = 'Inspector';
  document.getElementById('inspector-body').innerHTML = `
    <div id="inspector-empty">
      <div class="icon">⬡</div>
      <div>Click a node or edge<br>to inspect it here</div>
    </div>`;
}

function renderInspector(node) {
  const d    = node.data();
  const type = d.type;
  const col  = TYPE_COLOR[type] ?? '#64748b';

  document.getElementById('inspector-title').textContent = TYPE_LABEL[type] ?? type;
  document.getElementById('inspector-node-id').textContent = d.display_id;

  const badge = document.getElementById('inspector-type-badge');
  badge.style.display      = 'inline-block';
  badge.style.background   = col + '22';
  badge.style.color        = col;
  badge.style.border       = `1.5px solid ${col}55`;
  badge.textContent        = (TYPE_LABEL[type] ?? type).toUpperCase();

  const metaRows = [];
  if (d.paper_id)    metaRows.push(['Paper', d.paper_id]);
  if (d.section)     metaRows.push(['Section', d.section]);
  if (d.chunk_index != null) metaRows.push(['Chunk', `${d.chunk_index} / ${d.total_chunks ?? '?'}`]);
  if (d.score)       metaRows.push(['Score', d.score]);
  if (type === 'claim') {
    if (d.verdict)        metaRows.push(['Verdict', d.verdict]);
    if (d.verifier_used)  metaRows.push(['Verifier', d.verifier_used]);
    if (d.claim_type)     metaRows.push(['Claim type', d.claim_type]);
    if (d.hop_depth)      metaRows.push(['Hop depth', d.hop_depth]);
  }

  // Why-relevant block (shown for claim nodes, separately from the meta grid)
  const whyHtml = (type === 'claim' && d.why_relevant_to_question) ? `
    <div>
      <div class="section-title">Why relevant</div>
      <div class="inspector-text">${esc(d.why_relevant_to_question)}</div>
    </div>` : '';

  // Reason block (shown separately below meta grid for readability)
  const reasonHtml = (type === 'claim' && d.reason) ? `
    <div>
      <div class="section-title">Reason</div>
      <div class="inspector-text" style="font-style:italic;color:#94a3b8">${esc(d.reason)}</div>
    </div>` : '';

  const metaHtml = metaRows.length ? `
    <div>
      <div class="section-title">Metadata</div>
      <div class="meta-grid">
        ${metaRows.map(([k,v]) => `<div class="meta-key">${k}</div><div class="meta-val">${esc(v)}</div>`).join('')}
      </div>
    </div>` : '';

  let contentHtml = '';
  if (d.full_text) {
    if (type === 'claim') {
      const sqHtml = (Array.isArray(d.sub_query_texts) && d.sub_query_texts.length)
        ? `<div>
            <div class="section-title">Sub-queries</div>
            <div class="neighbor-list">
              ${d.sub_query_texts.map((t, i) => `
                <div class="neighbor-item">
                  <div class="neighbor-dot" style="background:#6366f1"></div>
                  <div class="neighbor-label">${esc(t)}</div>
                </div>`).join('')}
            </div>
          </div>`
        : '';
      contentHtml = `
        <div>
          <div class="section-title">Claim Content</div>
          <div class="claim-preview">
            <div class="claim-preview-text">${esc(d.full_text)}</div>
          </div>
        </div>
        ${sqHtml}`;
    } else {
      contentHtml = `
        <div>
          <div class="section-title">Content</div>
          <div class="inspector-text">${esc(d.full_text)}</div>
        </div>`;
    }
  }

  const neighbours = [];
  node.connectedEdges().forEach(e => {
    const other = e.source().id() === node.id() ? e.target() : e.source();
    if (other.id() === node.id()) return;
    neighbours.push({
      id:       other.id(),
      label:    other.data('label'),
      type:     other.data('type'),
      relation: e.data('relation'),
      dir:      e.source().id() === node.id() ? '→' : '←',
    });
  });

  const neighbourHtml = neighbours.length ? `
    <div>
      <div class="section-title">Connections (${neighbours.length})</div>
      <div class="neighbor-list">
        ${neighbours.map(n => `
          <div class="neighbor-item" onclick="focusNode('${esc(n.id)}')">
            <div class="neighbor-dot" style="background:${TYPE_COLOR[n.type] ?? '#64748b'}"></div>
            <div class="neighbor-label">${esc(n.label)}</div>
            <div class="neighbor-rel">${n.dir} ${esc(n.relation)}</div>
          </div>`).join('')}
      </div>
    </div>` : '';

  document.getElementById('inspector-body').innerHTML =
    metaHtml + whyHtml + reasonHtml + contentHtml + neighbourHtml;
}

function renderEdgeInspector(edge) {
  const d   = edge.data();
  const col = EDGE_COLOR[d.relation] ?? '#6366f1';

  document.getElementById('inspector-title').textContent = 'Edge';
  document.getElementById('inspector-node-id').textContent = `${d.source} → ${d.target}`;

  const badge = document.getElementById('inspector-type-badge');
  badge.style.display    = 'inline-block';
  badge.style.background = col + '22';
  badge.style.color      = col;
  badge.style.border     = `1.5px solid ${col}55`;
  badge.textContent      = d.relation.toUpperCase();

  document.getElementById('inspector-body').innerHTML = `
    <div>
      <div class="section-title">Metadata</div>
      <div class="meta-grid">
        <div class="meta-key">Relation</div><div class="meta-val">${esc(d.relation)}</div>
        <div class="meta-key">Score</div>   <div class="meta-val">${d.score}</div>
        <div class="meta-key">Source</div>  <div class="meta-val">${esc(d.source)}</div>
        <div class="meta-key">Target</div>  <div class="meta-val">${esc(d.target)}</div>
      </div>
    </div>`;
}

function pushBreadcrumb(id, label) {
  const existing = breadcrumbs.findIndex(c => c.id === id);
  if (existing !== -1) breadcrumbs = breadcrumbs.slice(0, existing + 1);
  else breadcrumbs.push({ id, label: label.replace(/…$/, '') });
  if (breadcrumbs.length > 6) breadcrumbs = breadcrumbs.slice(-6);
  renderBreadcrumb();
}

function renderBreadcrumb() {
  const el = document.getElementById('breadcrumb');
  if (breadcrumbs.length === 0) { el.innerHTML = ''; return; }
  el.innerHTML = breadcrumbs.map((c, i) => {
    const isCurrent = i === breadcrumbs.length - 1;
    return `${i > 0 ? '<span class="crumb-sep">›</span>' : ''}
      <span class="crumb ${isCurrent ? 'current' : ''}" onclick="focusNode('${esc(c.id)}')">${esc(c.label)}</span>`;
  }).join('');
}

window.focusNode = function(id) {
  const node = cy.getElementById(id);
  if (!node || node.empty()) return;
  cy.nodes().removeClass('selected');
  node.addClass('selected');
  selectedId = id;
  if (neighborMode) applyNeighborHighlight(id);
  cy.animate({ fit: { eles: node.closedNeighborhood(), padding: 90 } }, { duration: 350 });
  pushBreadcrumb(id, node.data('label'));
  renderInspector(node);
};

function esc(v) {
  return String(v ?? '')
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
const $ = (selector) => document.querySelector(selector);
const NAV = [
  ['overview', 'Overview'],
  ['graph', 'Knowledge Graph'],
  ['invest', 'Investigations'],
  ['gaps', 'Knowledge Gaps'],
  ['evidence', 'Evidence'],
  ['sources', 'Sources']
];
const STORAGE_KEY = 'nexus-demo-state-v1';
const MODE_KEY = 'nexus-mode-v1';
const API_BASE = window.NEXUS_API_BASE || 'http://127.0.0.1:8001';
const DEFAULT_STATUS = { type: 'info', message: '' };
let APP_MODE = window.localStorage.getItem(MODE_KEY) === 'demo' ? 'demo' : 'live';
const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (character) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
})[character]);

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch (error) {
    throw new Error(`Cannot reach the NEXSUS API at ${API_BASE}. Start the backend and try again.`);
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || `API request failed (${response.status}).`);
  }
  return payload;
}

function liveState() {
  return {
    ...INITIAL_STATE,
    status: { ...DEFAULT_STATUS },
    documents: [], evidence: [], relationships: [], gaps: [], investigations: [], concepts: [],
    activeInvestigation: null, activeInvestigationId: null, selectedGapId: null,
    pipelineStep: 0, knowledgeUpdated: false, counts: { documents: 0, concepts: 0, relationships: 0, gaps: 0, claims: 0 }
  };
}

async function refreshLiveState() {
  const data = await api('/state');
  S.documents = data.documents.map((doc) => ({
    ...doc,
    concepts: 0,
    relations: 0,
    evidence: doc.status === 'READY' ? doc.chunks : 0,
    demo: false
  }));
  S.concepts = data.graph.nodes.map((node) => ({ name: node.name, description: node.description }));
  S.relationships = data.graph.edges.map((edge) => ({
    source: edge.source, target: edge.target, type: edge.type, confidence: edge.confidence,
    status: edge.status.toLowerCase() === 'supported' ? 'supported' : edge.status.toLowerCase() === 'contradicted' ? 'contradiction' : 'weak',
    claim: edge.claim, sourceDoc: '', page: null
  }));
  S.gaps = data.gaps.map((gap) => ({
    id: gap.id, type: gap.type, description: gap.description, reason: gap.reason,
    relatedConcepts: gap.related_concepts, priority: gap.priority, coverage: '—', confidence: 0,
    status: gap.status
  }));
  S.evidence = data.evidence;
  S.investigations = data.investigations.map((item) => ({
    id: item.id, status: item.status === 'RUNNING' ? 'Running' : item.status === 'FAILED' ? 'Failed' : item.status.replaceAll('_', ' ').toLowerCase(),
    confidence: Number(item.confidence || 0), label: item.hypothesis || item.question || item.gap_id
  }));
  S.counts = data.counts;
  if (!S.activeInvestigationId && data.investigations.length) {
    S.activeInvestigationId = data.investigations[0].id;
  }
  if (S.activeInvestigationId) {
    const investigation = await api(`/investigations/${S.activeInvestigationId}`);
    S.activeInvestigation = {
      id: investigation.id, gap: investigation.gap, question: investigation.question,
      hypothesis: investigation.hypothesis, status: investigation.status, confidence: investigation.confidence,
      reasoningSummary: investigation.reasoning_summary, stage: investigation.stage,
      pipeline: ['Gap', 'Question', 'Retrieve', 'Hypothesis', 'Challenge', 'Verify', 'Update']
    };
    S.evidence = await api(`/investigations/${S.activeInvestigationId}/evidence`);
    const stages = { GAP: 0, QUESTION: 1, RETRIEVE: 2, HYPOTHESIS: 3, CHALLENGE: 4, VERIFY: 5, UPDATE: 6, COMPLETE: 7, FAILED: 0 };
    S.pipelineStep = stages[investigation.stage] ?? 0;
    S.knowledgeUpdated = investigation.stage === 'COMPLETE';
    if (!S.selectedGapId) S.selectedGapId = investigation.gap_id;
  }
  return data;
}

function safeReadStorage() {
  try {
    const item = window.localStorage.getItem(STORAGE_KEY);
    return item ? JSON.parse(item) : null;
  } catch (error) {
    return null;
  }
}

function persistStateSnapshot() {
  if (APP_MODE !== 'demo') return;
  try {
    const snapshot = {
      view: S.view,
      selectedConcept: S.selectedConcept,
      activeTab: S.activeTab,
      documents: S.documents,
      evidence: S.evidence,
      relationships: S.relationships,
      gaps: S.gaps,
      investigations: S.investigations,
      selectedGapId: S.selectedGapId,
      knowledgeUpdated: S.knowledgeUpdated,
      nextGapFound: S.nextGapFound,
      pipelineStep: S.pipelineStep,
      status: S.status
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot));
  } catch (error) {
    // Ignore storage failures; the app should continue without persistence.
  }
}

function setStatus(message, type = 'info') {
  S.status = { message, type };
  persistStateSnapshot();
}

function clearStatus() {
  S.status = { ...DEFAULT_STATUS };
  persistStateSnapshot();
}

const DEFINITION = {
  'Battery Degradation': { x: 300, y: 220 },
  'Thermal Stress': { x: 140, y: 110 },
  'Thermal Runaway': { x: 540, y: 110 },
  'Cycle Life': { x: 110, y: 320 },
  'Temperature': { x: 400, y: 380 },
  'Electrolyte Stability': { x: 650, y: 270 },
  'Charging Rate': { x: 70, y: 215 },
  'Lithium-Ion Battery': { x: 600, y: 410 }
};

const INITIAL_STATE = {
  view: 'overview',
  selectedConcept: 'Battery Degradation',
  activeTab: 'ALL',
  status: { ...DEFAULT_STATUS },
  documents: [
    { id: 'DOC-01', name: 'Battery Aging Review', pages: 42, status: 'Processed', concepts: 24, relations: 31, evidence: 12, demo: true },
    { id: 'DOC-02', name: 'Thermal Abuse Casebook', pages: 18, status: 'Processed', concepts: 19, relations: 28, evidence: 9, demo: true },
    { id: 'DOC-03', name: 'Cell Chemistry Summary', pages: 67, status: 'Queued', concepts: 12, relations: 14, evidence: 3, demo: true }
  ],
  evidence: [
    {
      source: 'Paper_03.pdf',
      page: 9,
      text: 'Aged cells showed lower onset temperature for self-heating and earlier instability under elevated temperature.',
      relevance: '0.91',
      type: 'supporting',
      relationship: 'Battery Degradation → Thermal Runaway',
      investigationId: 'INV-007',
      contradiction: false
    },
    {
      source: 'Paper_07.pdf',
      page: 14,
      text: 'Capacity fade above 50 °C coincided with earlier thermal instability, suggesting degradation accelerates safety risk.',
      relevance: '0.84',
      type: 'supporting',
      relationship: 'Battery Degradation → Thermal Runaway',
      investigationId: 'INV-007',
      contradiction: false
    },
    {
      source: 'Study_09.pdf',
      page: 31,
      text: 'Lithium plating in degraded anodes lowered the abuse-test threshold, increasing the chance of runaway onset.',
      relevance: '0.77',
      type: 'supporting',
      relationship: 'Battery Degradation → Thermal Runaway',
      investigationId: 'INV-007',
      contradiction: false
    },
    {
      source: 'Paper_11.pdf',
      page: 22,
      text: 'No change in runaway onset was observed after 500 cycles at 45 °C, suggesting the causality may depend on temperature threshold.',
      relevance: 'Direct contradiction',
      type: 'counter',
      relationship: 'Battery Degradation → Thermal Runaway',
      investigationId: 'INV-007',
      contradiction: true
    },
    {
      source: 'Report_14.pdf',
      page: 5,
      text: 'Thermal runaway onset was governed more strongly by separator quality and local shorting conditions than by cell age alone.',
      relevance: 'Alternative cause',
      type: 'counter',
      relationship: 'Battery Degradation → Thermal Runaway',
      investigationId: 'INV-007',
      contradiction: true
    }
  ],
  relationships: [
    { source: 'Temperature', target: 'Battery Degradation', type: 'accelerates', confidence: 0.82, status: 'supported', claim: 'Temperature increases degradation risk.', sourceDoc: 'Paper_03.pdf', page: 9 },
    { source: 'Battery Degradation', target: 'Cycle Life', type: 'reduces', confidence: 0.8, status: 'supported', claim: 'Battery degradation shortens cycle life.', sourceDoc: 'Paper_02.pdf', page: 18 },
    { source: 'Battery Degradation', target: 'Thermal Stress', type: 'increases', confidence: 0.68, status: 'weak', claim: 'Aged cells produce more thermal stress.', sourceDoc: 'Study_04.pdf', page: 11 },
    { source: 'Thermal Stress', target: 'Thermal Runaway', type: 'increases', confidence: 0.71, status: 'weak', claim: 'Thermal stress increases risk of runaway.', sourceDoc: 'Paper_08.pdf', page: 6 },
    { source: 'Charging Rate', target: 'Battery Degradation', type: 'influences', confidence: 0.64, status: 'weak', claim: 'Charging rate modulates degradation acceleration.', sourceDoc: 'Report_05.pdf', page: 17 },
    { source: 'Lithium-Ion Battery', target: 'Electrolyte Stability', type: 'depends on', confidence: 0.59, status: 'weak', claim: 'Chemistry and electrolyte stability shape cell durability.', sourceDoc: 'Research_Report.pdf', page: 29 }
  ],
  gaps: [
    {
      id: 'GAP-007',
      type: 'Missing Relationship',
      description: 'Battery Degradation → ??? → Thermal Runaway',
      reason: 'Multiple sources discuss aging and thermal runaway, but no validated relationship connects them under elevated-temperature conditions.',
      relatedConcepts: ['Battery Degradation', 'Thermal Runaway'],
      priority: 'High priority',
      coverage: '42%',
      confidence: 0.81,
      status: 'Open'
    },
    {
      id: 'GAP-003',
      type: 'Contradiction',
      description: 'Evidence conflicts on whether degradation changes runaway onset.',
      reason: 'One study showed earlier instability while another showed no measurable shift at 45 °C.',
      relatedConcepts: ['Battery Degradation', 'Thermal Runaway', 'Temperature'],
      priority: 'Moderate',
      coverage: '58%',
      confidence: 0.61,
      status: 'Open'
    }
  ],
  investigations: [
    { id: 'INV-006', status: 'Verified', confidence: 0.91, label: 'Temperature → Battery Degradation' },
    { id: 'INV-005', status: 'Contradicted', confidence: 0.34, label: 'Charge rate mismatch' },
    { id: 'INV-007', status: 'Partially supported', confidence: 0.68, label: 'Battery Degradation → Thermal Runaway' }
  ],
  activeInvestigation: null,
  pipelineStep: 0,
  pipelineTimer: null,
  knowledgeUpdated: false,
  nextGapFound: false,
  selectedGapId: 'GAP-007'
};

const cloneState = (value) => typeof structuredClone === 'function' ? structuredClone(value) : JSON.parse(JSON.stringify(value));
const restoredState = APP_MODE === 'demo' ? safeReadStorage() : null;
const S = APP_MODE === 'demo' ? cloneState(INITIAL_STATE) : liveState();
if (APP_MODE === 'demo') {
  S.concepts = Object.keys(DEFINITION).map((name) => ({ name, description: '' }));
}
if (restoredState && APP_MODE === 'demo') {
  Object.assign(S, restoredState);
  if (!S.status) {
    S.status = { ...DEFAULT_STATUS };
  }
}

function stat(label, value, color = '') {
  return `<div class="summary-card">
    <div class="muted small">${label}</div>
    <div class="number" ${color ? `style="color:${color}"` : ''}>${value}</div>
  </div>`;
}

function head(title, subtitle = '', actions = '') {
  return `<div class="header-row"><div><h1>${title}</h1>${subtitle ? `<div class="muted" style="margin-top:4px">${subtitle}</div>` : ''}</div><div>${actions}</div></div>`;
}

function badge(type, text) {
  return `<span class="badge ${type}">${text}</span>`;
}

function getActiveGap() {
  return S.gaps.find((gap) => gap.id === S.selectedGapId) || S.gaps[0];
}

function getQuestionForGap(gap) {
  if (gap && gap.type === 'Contradiction') {
    return 'Which temperature window determines whether degradation changes runaway onset?';
  }
  return 'Does battery degradation increase thermal runaway risk under elevated-temperature conditions?';
}

function graphView(height = 420, compact = false) {
  const concepts = APP_MODE === 'demo' ? Object.keys(DEFINITION) : S.concepts.map((concept) => concept.name);
  if (!concepts.length) {
    return '<div class="muted-panel">No extracted concepts yet. Upload and process a PDF to build the graph.</div>';
  }
  const positions = APP_MODE === 'demo'
    ? DEFINITION
    : Object.fromEntries(concepts.map((name, index) => {
        const angle = concepts.length ? (2 * Math.PI * index / concepts.length) - Math.PI / 2 : 0;
        return [name, { x: concepts.length === 1 ? 380 : 380 + 265 * Math.cos(angle), y: concepts.length === 1 ? 230 : 230 + 165 * Math.sin(angle) }];
      }));
  const visibleRelations = APP_MODE === 'demo' && S.knowledgeUpdated
    ? [
        ...S.relationships,
        { source: 'Battery Degradation', target: 'Thermal Runaway', type: 'may increase', confidence: 0.68, status: 'weak', sourceDoc: 'INV-007', page: null }
      ]
    : S.relationships;

  const edgeMarkup = visibleRelations
    .map((rel) => {
      if (!positions[rel.source] || !positions[rel.target]) {
        return '';
      }

      const [x1, y1] = [positions[rel.source].x, positions[rel.source].y];
      const [x2, y2] = [positions[rel.target].x, positions[rel.target].y];
      const mx = (x1 + x2) / 2;
      const my = (y1 + y2) / 2;
      const dx = x2 - x1;
      const dy = y2 - y1;
      const distance = Math.max(Math.hypot(dx, dy), 1);
      const ux = dx / distance;
      const uy = dy / distance;
      const isUnknown = rel.status === 'unknown';
      const strokeColor = rel.status === 'supported' ? '#34d399' : rel.status === 'contradiction' ? '#fb7185' : rel.status === 'weak' ? '#fbbf24' : '#94a3b8';
      const label = rel.status === 'unknown' ? '???' : rel.type;
      const strokeWidth = rel.status === 'unknown' ? 2.6 : 1.8;
      const dash = rel.status === 'unknown' ? 'stroke-dasharray="8 5"' : '';
      const textColor = rel.status === 'unknown' ? '#94a3b8' : '#cbd5e1';

      return `
        <line class="graph-edge${isUnknown ? ' graph-edge-unknown' : ''}" ${dash} x1="${x1 + ux * 40}" y1="${y1 + uy * 20}" x2="${x2 - ux * 40}" y2="${y2 - uy * 20}" stroke="${strokeColor}" stroke-width="${strokeWidth}" marker-end="url(#edgeArrow)" />
        <text x="${mx}" y="${my - 6}" text-anchor="middle" style="fill:${textColor};font-size:${rel.status === 'unknown' ? 14 : 11}px;font-weight:${rel.status === 'unknown' ? 700 : 500}">${escapeHtml(label)}</text>
      `;
    })
    .join('');

  const nodes = Object.entries(positions)
    .map(([name, pos]) => {
      const width = Math.max(120, name.length * 7 + 22);
      const selected = S.selectedConcept === name && !compact;
      return `<g class="node ${selected ? 'selected' : ''}" data-node="${escapeHtml(name)}" transform="translate(${pos.x - width / 2}, ${pos.y - 18})">
        <rect width="${width}" height="36" rx="7" />
        <text x="${width / 2}" y="22" text-anchor="middle">${escapeHtml(name)}</text>
      </g>`;
    })
    .join('');

  return `<svg viewBox="0 0 760 460" style="width:100%;height:${height}px;display:block">
    <defs>
      <marker id="edgeArrow" viewBox="0 0 8 8" refX="7.5" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L8 4L0 8z" fill="#4f7cff"/></marker>
    </defs>
    ${edgeMarkup}
    ${nodes}
  </svg>`;
}

function renderEvidenceCard(item) {
  const tone = item.contradiction ? 'counter' : 'support';
  const label = item.contradiction ? 'Counter-evidence' : item.type === 'source' ? 'Source chunk' : 'Supporting evidence';
  return `<div class="evidence-card ${tone}" data-open-evidence="${escapeHtml(item.source)}">
    <div class="row compact">
      <strong>${escapeHtml(item.source)}</strong>
      <span class="muted small">p.${escapeHtml(item.page)}</span>
    </div>
    <p>“${escapeHtml(item.text)}”</p>
    <div class="row compact footer">
      <span class="badge ${item.contradiction ? 'danger' : 'success'}">${label}</span>
      <span class="muted small">${escapeHtml(item.relevance)}</span>
    </div>
  </div>`;
}

function renderGapCard(gap) {
  return `<div class="card gap-panel interactive-card">
    <div class="row compact">
      <strong class="danger-text">${escapeHtml(gap.type)}</strong>
      ${badge('secondary', escapeHtml(gap.priority))}
    </div>
    <div class="gap-header">${escapeHtml(gap.description)}</div>
    <div class="muted small" style="margin-top:8px">Why detected</div>
    <p>${escapeHtml(gap.reason)}</p>
    <div class="row compact">
      <span class="muted small">Coverage ${escapeHtml(gap.coverage || '—')}</span>
      <span class="muted small">Confidence ${gap.confidence ? `${(gap.confidence * 100).toFixed(0)}%` : '—'}</span>
    </div>
    <div class="row compact actions">
        <button class="btn secondary" data-action="open-gap" data-gap-id="${escapeHtml(gap.id)}">View gap</button>
      <button class="btn" data-action="investigate-gap" data-gap-id="${escapeHtml(gap.id)}">Investigate</button>
    </div>
  </div>`;
}

function renderOverview() {
  const activeGap = getActiveGap();
  const counts = S.counts || {};
  const recentEvents = S.investigations.slice(0, 5).map((inv) => [inv.id.startsWith('INV-') ? inv.id : inv.id.slice(0, 8), inv.label || inv.status]);
  return head('Overview', 'NEXUS continuously scans for what the knowledge base has not asked yet.', `<button class="btn big" data-action="start-investigation">Start autonomous investigation</button>`) + `
    <div class="stats-grid">
      ${stat('Documents', counts.documents ?? S.documents.length)}
      ${stat('Concepts', counts.concepts ?? S.concepts.length)}
      ${stat('Relationships', counts.relationships ?? S.relationships.length)}
      ${stat('Open gaps', counts.gaps ?? S.gaps.length, '#4f7cff')}
    </div>
    <div class="content-grid two-col">
      <div class="card graph-card">
        <div class="section-header">
          <h2>Knowledge graph</h2>
        </div>
        ${graphView(340, true)}
        <div class="muted small" style="margin-top:12px; display:flex; gap:18px; flex-wrap:wrap;">
          <span style="color:var(--ok)">supports</span>
          <span style="color:var(--bad)">contradiction</span>
          <span style="color:var(--wa)">weak</span>
          <span style="color:#94a3b8">unknown</span>
        </div>
        <div style="margin-top:12px"><button class="btn secondary" data-action="go-graph">Open full graph</button></div>
      </div>
      <div class="side-stack">
        <div class="card">
          <div class="section-header"><h2>Knowledge intelligence</h2></div>
          <div class="list-row"><span>Open gaps</span><span class="badge accent">${S.gaps.length}</span></div>
          <div class="list-row"><span>Contradictions</span><span class="badge danger">${S.gaps.filter((gap) => gap.type === 'Contradiction').length}</span></div>
          <div class="list-row"><span>Weak evidence</span><span class="badge warning">${S.relationships.filter((rel) => rel.status === 'weak').length}</span></div>
          <div class="list-row"><span>Unverified claims</span><span class="badge muted-badge">${APP_MODE === 'demo' ? 2 : counts.claims ?? 0}</span></div>
        </div>
        <div class="card">
          <div class="section-header"><h2>Recent investigations</h2></div>
          ${S.investigations.length ? S.investigations.map((inv) => `<div class="list-row"><span>#${escapeHtml(inv.id.startsWith('INV-') ? inv.id.split('-')[1] : inv.id.slice(0, 8))} <span class="badge ${inv.status === 'Verified' || inv.status === 'supported' ? 'success' : inv.status === 'Contradicted' || inv.status === 'contradicted' ? 'danger' : 'warning'}">${escapeHtml(inv.status)}</span></span><span class="muted small">${Math.round(inv.confidence * 100)}%</span></div>`).join('') : '<div class="muted-panel">No investigations yet.</div>'}
        </div>
      </div>
    </div>
    <div class="card timeline-card" style="margin-top:16px">
      <div class="section-header"><h2>Activity</h2></div>
      <div class="timeline">
        ${recentEvents.length ? recentEvents.map(([id, event]) => `<div class="timeline-item"><span class="muted small">${escapeHtml(id)}</span> <span>${escapeHtml(event)}</span></div>`).join('') : '<div class="muted-panel">Activity will appear after processing a document.</div>'}
      </div>
    </div>
    ${APP_MODE === 'demo' && activeGap ? `<div class="card demo-banner">
      <strong>DEMO DATA</strong>
      <span>${escapeHtml(activeGap.description)}</span>
    </div>` : ''}
  `;
}

function renderGraph() {
  const selected = S.concepts.some((concept) => concept.name === S.selectedConcept)
    ? S.selectedConcept
    : S.concepts[0]?.name || 'No concepts extracted';
  const currentGap = getActiveGap();
  const connected = S.relationships.filter((rel) => rel.source === selected || rel.target === selected);
  const meanConfidence = connected.length ? Math.round(connected.reduce((sum, rel) => sum + rel.confidence, 0) / connected.length * 100) : null;
  return head('Knowledge graph', `Structured state: ${S.concepts.length} concepts · ${S.relationships.length} relationships · ${S.gaps.length} open gaps`, `
    <input class="search-box" value="${selected}" readonly>
    <button class="btn secondary">Filter</button>
  `) + (S.knowledgeUpdated ? `
    <div class="card success-box row compact">
      <div><strong>Knowledge updated.</strong> Battery Degradation may increase Thermal Runaway under elevated temperature conditions.</div>
      <span class="badge accent">1 new gap found</span>
    </div>
  ` : '') + `
    <div class="content-grid two-col graph-layout">
      <div class="card graph-card">${graphView(520, false)}</div>
      <div class="card details-card">
        <div class="muted small">Concept</div>
        <h2>${escapeHtml(selected)}</h2>
        <div class="list-row"><span class="muted">Mentions</span><span>${APP_MODE === 'demo' ? '14 sources' : 'See linked evidence'}</span></div>
        <div class="list-row"><span class="muted">Relationships</span><span>${connected.length}</span></div>
        <div class="list-row"><span class="muted">Confidence</span><span>${meanConfidence === null ? '—' : `${meanConfidence}%`}</span></div>
        <div class="list-row"><span class="muted">Sources</span><span>${APP_MODE === 'demo' ? '03 · 07 · 11' : 'Evidence view'}</span></div>
        <div class="list-row"><span class="muted">Open questions</span><span>${selected === 'Battery Degradation' || selected === 'Thermal Runaway' ? '2' : '0'}</span></div>
        <button class="btn" style="width:100%;margin-top:16px" data-action="start-investigation" ${S.gaps.length ? '' : 'disabled'}>Investigate</button>
      </div>
    </div>
    <div class="card insight-card" style="margin-top:16px">
      <strong>${escapeHtml(currentGap?.description || 'No open knowledge gaps')}</strong>
      <p>${escapeHtml(currentGap?.reason || 'Upload source documents to discover evidence-grounded gaps.')}</p>
    </div>
  `;
}

function renderGaps() {
  const tabs = ['ALL', 'MISSING RELATIONSHIP', 'CONTRADICTION', 'WEAK EVIDENCE', 'UNVERIFIED'];
  const filtered = S.activeTab === 'ALL'
    ? S.gaps
    : S.gaps.filter((gap) => gap.type.toUpperCase().replace(/\s+/g, '_') === S.activeTab.replace(/\s+/g, '_'));

  return head('Knowledge gaps', 'Unresolved relationships and unsupported claims discovered by NEXUS.') + `
    <div class="tabs">
      ${tabs.map((tab) => `<button class="tab ${S.activeTab === tab ? 'active' : ''}" data-tab="${tab}">${tab}</button>`).join('')}
    </div>
    <div class="gap-list">
      ${filtered.length ? filtered.map(renderGapCard).join('') : '<div class="card muted-panel">No gaps in this category.</div>'}
    </div>
  `;
}

function renderGapDetail() {
  const gap = getActiveGap();
  if (!gap) return head('Knowledge gap', 'No open gaps are available.') + '<div class="muted-panel">Process source documents to detect knowledge gaps.</div>';
  return head('Knowledge gap', `${gap.id} · ${gap.type}`) + `
    <div class="card gap-detail">
      <div class="detail-header">${escapeHtml(gap.description)}</div>
      <div class="row compact" style="margin-top:12px">
        <span class="muted small">Confidence ${(gap.confidence * 100).toFixed(0)}%</span>
        <span class="badge accent">${gap.priority}</span>
      </div>
      <p><strong>Why detected:</strong> ${escapeHtml(gap.reason)}</p>
      <p><strong>Concepts:</strong> ${escapeHtml((gap.relatedConcepts || []).join(' · ') || 'Not linked to named concepts')}</p>
      <p><strong>Evidence coverage:</strong> ${escapeHtml(gap.coverage || 'Not calculated')}</p>
      <div class="row compact actions">
        <button class="btn secondary" data-action="go-gaps">Back to gaps</button>
        <button class="btn" data-action="investigate-gap" data-gap-id="${gap.id}">Investigate</button>
      </div>
    </div>
  `;
}

function renderEvidence() {
  const preview = S.evidence[0];
  return head('Evidence', 'Every important claim is labelled with source, page, and provenance.') + `
    <div class="content-grid two-col evidence-layout">
      <div class="card">
        <div class="section-header"><h2>Evidence stream</h2></div>
        <div class="evidence-list">
          ${S.evidence.map(renderEvidenceCard).join('')}
        </div>
      </div>
      <div class="card">
        <div class="section-header"><h2>Source preview</h2></div>
        ${preview ? `<div class="muted small">${escapeHtml(preview.source)} · page ${escapeHtml(preview.page)}</div><p style="margin-top:12px">${escapeHtml(preview.text)}</p>` : '<div class="muted-panel">No source evidence has been ingested yet.</div>'}
      </div>
    </div>
  `;
}

function renderSources() {
  return head('Sources', 'Uploaded documents and processing state.', '<button class="btn secondary" id="upload-button">+ Add sources</button>') + `
    <input id="document-upload" type="file" accept=".pdf,application/pdf" multiple style="display:none" />
    <div class="source-grid">
      ${S.documents.map((doc) => {
        const isProcessing = ['UPLOADING', 'PROCESSING', 'PARSING'].includes(String(doc.status).toUpperCase());
        return `
        <div class="card source-card">
          <div class="row compact">
            <strong>${escapeHtml(doc.name)}</strong>
              <span class="badge ${['READY', 'Processed'].includes(doc.status) ? 'success' : doc.status === 'FAILED' ? 'danger' : 'accent'}${isProcessing ? ' is-processing' : ''}">${escapeHtml(doc.status)}</span>
          </div>
            <div class="muted small">${escapeHtml(doc.pages || 0)} pages${doc.demo ? ' · DEMO DATA' : ''}${doc.error ? ` · ${escapeHtml(doc.error)}` : ''}</div>
          <div class="list-row"><span class="muted">Concepts</span><span>${doc.concepts}</span></div>
          <div class="list-row"><span class="muted">Relationships</span><span>${doc.relations}</span></div>
          <div class="list-row"><span class="muted">Evidence</span><span>${doc.evidence}</span></div>
          ${!doc.demo && doc.status === 'FAILED' ? `<div class="row compact actions"><button class="btn secondary" data-action="retry-document" data-document-id="${escapeHtml(doc.id)}">Retry processing</button></div>` : ''}
        </div>
      `;
      }).join('')}
    </div>
  `;
}

function renderInvestigation() {
  if (APP_MODE === 'live' && !S.activeInvestigation) {
    return head('Investigation', 'No investigation is currently selected.') +
      `<div class="muted-panel">${S.gaps.length ? 'Start an investigation from an open evidence-grounded knowledge gap.' : 'Process source documents to generate knowledge gaps before investigating.'}</div>`;
  }
  const investigation = S.activeInvestigation || {
    id: 'INV-007',
    gap: getActiveGap().description,
    question: getQuestionForGap(getActiveGap()),
    status: 'PARTIALLY_SUPPORTED',
    confidence: 68,
    pipeline: ['Gap', 'Question', 'Retrieve', 'Hypothesis', 'Challenge', 'Verify', 'Update']
  };

  const activeStage = Math.min(S.pipelineStep, investigation.pipeline.length - 1);
  const evidenceSummary = S.evidence.filter((item) => item.type === 'supporting' || (APP_MODE === 'demo' && !item.contradiction));
  const counterSummary = S.evidence.filter((item) => item.type === 'counter' || (APP_MODE === 'demo' && item.contradiction));
  const verificationStatus = investigation.status || 'UNVERIFIED';
  const confidence = APP_MODE === 'demo' ? 68 : Math.round((investigation.confidence || 0) * 100);
  const completed = S.pipelineStep >= 7;
  const isRunning = APP_MODE === 'demo'
    ? Boolean(S.activeInvestigation) && S.pipelineStep < 7 && S.status.type === 'info'
    : investigation.status === 'RUNNING';

  return head('Investigation', `${escapeHtml(investigation.id)} · ${escapeHtml(investigation.gap)}`, `<span class="badge ${S.pipelineStep >= 7 ? 'success' : 'accent'}">${S.pipelineStep >= 7 ? 'Complete' : 'Running'}</span>`) + `
    <div class="card investigation-card${isRunning ? ' is-running' : ''}">
      <div class="muted small">Current question</div>
      <h2>“${escapeHtml(investigation.question || 'Generating research question from the selected gap…')}”</h2>
      <div class="pipeline">
        ${investigation.pipeline.map((step, index) => `<div class="pipeline-step ${index < S.pipelineStep ? 'done' : index === S.pipelineStep ? 'active' : ''}">${escapeHtml(step)}</div>`).join('')}
      </div>
      <div class="timeline compact-timeline">
        ${[
          'GAP DETECTED',
          'QUESTION GENERATED',
          'SOURCES RETRIEVED',
          'EVIDENCE EXTRACTED',
          'HYPOTHESIS FORMED',
          'COUNTER-EVIDENCE SEARCHED',
          'VERIFICATION COMPLETED',
          'KNOWLEDGE GRAPH UPDATED'
        ].map((label, index) => `<div class="timeline-item ${index <= activeStage ? 'done' : ''}">${label}</div>`).join('')}
      </div>
    </div>
    <div class="content-grid two-col timer-grid">
      <div class="card">
        <div class="section-header"><h2>Supporting evidence</h2></div>
        <div class="evidence-list">
          ${evidenceSummary.length ? evidenceSummary.map(renderEvidenceCard).join('') : `<div class="muted-panel">${completed ? 'Insufficient evidence found in the uploaded knowledge base.' : 'Supporting evidence will appear after retrieval.'}</div>`}
        </div>
      </div>
      <div class="card">
        <div class="section-header"><h2>Counter-evidence</h2></div>
        <div class="evidence-list">
          ${counterSummary.length ? counterSummary.map(renderEvidenceCard).join('') : `<div class="muted-panel">${completed ? 'No counter-evidence was selected from retrieved source chunks.' : 'Counter-evidence will appear after the challenger search.'}</div>`}
        </div>
      </div>
    </div>
    ${S.pipelineStep >= 6 ? `
      <div class="card verification-panel">
        <div class="row compact">
          <h1 class="warning-text">${escapeHtml(verificationStatus.replaceAll('_', ' ').toLowerCase())}</h1>
          <div class="number small-number">${confidence}% <span class="muted small">confidence</span></div>
        </div>
        <div class="meter">
          <span style="width:${confidence}%;background:var(--ok)"></span>
          <span style="width:${100 - confidence}%;background:var(--wa)"></span>
        </div>
        <p><strong>Hypothesis:</strong> ${escapeHtml(investigation.hypothesis || 'No evidence-grounded hypothesis was returned.')}</p>
        <p class="muted">${escapeHtml(investigation.reasoningSummary || (completed ? 'Insufficient evidence found in the uploaded knowledge base.' : 'Verification is being finalized.'))}</p>
        <div style="margin-top:14px">
          <button class="btn big" data-action="update-graph">${APP_MODE === 'demo' ? 'Update knowledge graph' : 'Refresh knowledge graph'}</button>
        </div>
      </div>
    ` : ''}
  `;
}

function renderCurrentScreen() {
  if (S.view === 'overview') return renderOverview();
  if (S.view === 'graph') return renderGraph();
  if (S.view === 'gaps') return renderGaps();
  if (S.view === 'gap') return renderGapDetail();
  if (S.view === 'evidence') return renderEvidence();
  if (S.view === 'sources') return renderSources();
  if (S.view === 'invest') return renderInvestigation();
  return renderOverview();
}

function renderStatusBanner() {
  if (!S.status || !S.status.message) {
    return '';
  }

  const typeClass = S.status.type === 'error' ? 'status-error' : S.status.type === 'success' ? 'status-success' : 'status-info';
  return `<div class="card status-banner ${typeClass}">${escapeHtml(S.status.message)}</div>`;
}

function updateNav() {
  $('#nav').innerHTML = NAV.map(([key, label]) => `<button class="nav-item ${S.view === key ? 'on' : ''}" data-go="${key}">${label}</button>`).join('');
  const modeButton = $('#mode-switch');
  modeButton.textContent = APP_MODE === 'demo' ? 'DEMO MODE · SWITCH TO LIVE' : 'LIVE KNOWLEDGE MODE · SWITCH TO DEMO';
  modeButton.classList.toggle('demo', APP_MODE === 'demo');
  modeButton.onclick = toggleMode;
}

async function toggleMode() {
  APP_MODE = APP_MODE === 'demo' ? 'live' : 'demo';
  window.localStorage.setItem(MODE_KEY, APP_MODE);
  const next = APP_MODE === 'demo' ? cloneState(INITIAL_STATE) : liveState();
  if (APP_MODE === 'demo') {
    next.concepts = Object.keys(DEFINITION).map((name) => ({ name, description: '' }));
    const saved = safeReadStorage();
    if (saved) Object.assign(next, saved);
  }
  Object.assign(S, next);
  S.activeInvestigationId = null;
  if (APP_MODE === 'live') {
    setStatus('Connecting to the live knowledge service...', 'info');
    render();
    try {
      await refreshLiveState();
      clearStatus();
    } catch (error) {
      setStatus(error.message, 'error');
    }
  } else {
    clearStatus();
  }
  render();
}

async function startLiveInvestigation() {
  try {
    S.view = 'invest';
    setStatus('Selecting an open corpus gap...', 'info');
    render();
    const gapQuery = S.selectedGapId ? `?gap_id=${encodeURIComponent(S.selectedGapId)}` : '';
    const investigation = await api(`/investigations/start${gapQuery}`, { method: 'POST' });
    S.activeInvestigationId = investigation.id;
    await api(`/investigations/${investigation.id}/run`, { method: 'POST' });
    await pollLiveInvestigation(investigation.id);
  } catch (error) {
    setStatus(error.message, 'error');
    render();
  }
}

async function pollLiveInvestigation(investigationId) {
  const messages = {
    GAP: 'Selected an open gap from the live knowledge graph.', QUESTION: 'Generating a focused research question.',
    RETRIEVE: 'Retrieving relevant source chunks.', HYPOTHESIS: 'Forming a hypothesis from retrieved evidence.',
    CHALLENGE: 'Searching specifically for counter-evidence.', VERIFY: 'Comparing retrieved support and counter-evidence.',
    UPDATE: 'Updating the graph and detecting the next gap.'
  };
  for (let attempt = 0; attempt < 900; attempt += 1) {
    const investigation = await api(`/investigations/${investigationId}`);
    S.activeInvestigation = {
      id: investigation.id, gap: investigation.gap, question: investigation.question,
      hypothesis: investigation.hypothesis, status: investigation.status, confidence: investigation.confidence,
      reasoningSummary: investigation.reasoning_summary, stage: investigation.stage,
      pipeline: ['Gap', 'Question', 'Retrieve', 'Hypothesis', 'Challenge', 'Verify', 'Update']
    };
    const stages = { GAP: 0, QUESTION: 1, RETRIEVE: 2, HYPOTHESIS: 3, CHALLENGE: 4, VERIFY: 5, UPDATE: 6, COMPLETE: 7, FAILED: 0 };
    S.pipelineStep = stages[investigation.stage] ?? 0;
    S.evidence = await api(`/investigations/${investigationId}/evidence`);
    setStatus(investigation.status === 'FAILED' ? investigation.reasoning_summary : messages[investigation.stage] || 'Investigation complete.', investigation.status === 'FAILED' ? 'error' : investigation.stage === 'COMPLETE' ? 'success' : 'info');
    render();
    if (investigation.status === 'FAILED' || investigation.stage === 'FAILED') return;
    if (investigation.stage === 'COMPLETE') {
      await refreshLiveState();
      setStatus('Investigation verified; graph updated and gap detection rerun.', 'success');
      render();
      return;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 800));
  }
  throw new Error('Investigation is still running. Reopen the Investigation view to check its persisted status.');
}

async function uploadLiveFiles(files) {
  S.view = 'sources';
  for (const file of files) {
    if (!file.name.toLowerCase().endsWith('.pdf')) throw new Error(`${file.name}: only PDF documents are accepted.`);
    if (file.size <= 0) throw new Error(`${file.name}: the selected document is empty.`);
    setStatus(`Extracting and processing ${file.name}...`, 'info');
    render();
    const form = new FormData();
    form.append('file', file);
    const document = await api('/documents/upload', { method: 'POST', body: form });
    for (let attempt = 0; attempt < 900; attempt += 1) {
      const current = await api(`/documents/${document.id}`);
      if (current.status === 'FAILED') throw new Error(`${current.name}: ${current.error || 'Document processing failed.'}`);
      if (current.status === 'READY') break;
      setStatus(`${current.status === 'UPLOADING' ? 'Uploading' : 'Extracting, embedding, and analyzing'} ${current.name}...`, 'info');
      render();
      if (attempt === 899) throw new Error(`${current.name}: processing has not completed yet. Check Sources for its status.`);
      await new Promise((resolve) => window.setTimeout(resolve, 800));
    }
  }
  await refreshLiveState();
  setStatus(`${files.length} PDF${files.length === 1 ? '' : 's'} processed from source text.`, 'success');
  render();
}

async function retryLiveDocument(documentId) {
  S.view = 'sources';
  setStatus('Retrying document extraction from its persisted chunks...', 'info');
  render();
  try {
    await api(`/documents/${encodeURIComponent(documentId)}/process`, { method: 'POST' });
    for (let attempt = 0; attempt < 900; attempt += 1) {
      const current = await api(`/documents/${encodeURIComponent(documentId)}`);
      if (current.status === 'FAILED') throw new Error(`${current.name}: ${current.error || 'Document processing failed.'}`);
      if (current.status === 'READY') break;
      setStatus(`Retrying extraction for ${current.name} from stored chunks...`, 'info');
      render();
      if (attempt === 899) throw new Error(`${current.name}: processing has not completed yet. Check Sources for its status.`);
      await new Promise((resolve) => window.setTimeout(resolve, 800));
    }
    await refreshLiveState();
    setStatus('Document extraction completed from persisted source chunks.', 'success');
  } catch (error) {
    try { await refreshLiveState(); } catch (_) {}
    setStatus(error.message, 'error');
  }
  render();
}

function bindEvents() {
  document.querySelectorAll('[data-go]').forEach((button) => {
    button.addEventListener('click', () => {
      const view = button.dataset.go;
      if (view === 'overview') S.view = 'overview';
      else if (view === 'graph') S.view = 'graph';
      else if (view === 'invest') S.view = 'invest';
      else if (view === 'gaps') S.view = 'gaps';
      else if (view === 'evidence') S.view = 'evidence';
      else if (view === 'sources') S.view = 'sources';
      render();
    });
  });

  document.querySelectorAll('[data-tab]').forEach((button) => {
    button.addEventListener('click', () => {
      S.activeTab = button.dataset.tab;
      render();
    });
  });

  document.querySelectorAll('.node').forEach((node) => {
    node.addEventListener('click', () => {
      const concept = node.dataset.node;
      if (concept) {
        S.selectedConcept = concept;
        S.view = 'graph';
        render();
      }
    });
  });

  document.querySelectorAll('[data-action]').forEach((button) => {
    const action = button.dataset.action;
    button.addEventListener('click', () => {
      if (action === 'start-investigation') {
        if (APP_MODE === 'demo') startInvestigation();
        else startLiveInvestigation();
      }
      if (action === 'go-graph') {
        S.view = 'graph';
        render();
      }
      if (action === 'go-gaps') {
        S.view = 'gaps';
        render();
      }
      if (action === 'open-gap') {
        S.selectedGapId = button.dataset.gapId;
        S.view = 'gap';
        render();
      }
      if (action === 'investigate-gap') {
        S.selectedGapId = button.dataset.gapId || S.selectedGapId;
        if (APP_MODE === 'demo') startInvestigation();
        else startLiveInvestigation();
      }
      if (action === 'update-graph') {
        if (APP_MODE === 'demo') updateKnowledgeGraph();
        else api('/knowledge-graph/update', { method: 'POST' }).then(async () => {
          await refreshLiveState();
          S.view = 'graph';
          setStatus('Knowledge graph refreshed from persisted evidence.', 'success');
          render();
        }).catch((error) => { setStatus(error.message, 'error'); render(); });
      }
      if (action === 'retry-document' && APP_MODE === 'live') {
        retryLiveDocument(button.dataset.documentId);
      }
    });
  });

  const uploadButton = document.getElementById('upload-button');
  const uploadInput = document.getElementById('document-upload');
  if (uploadButton && uploadInput) {
    uploadButton.addEventListener('click', () => uploadInput.click());
    uploadInput.addEventListener('change', async (event) => {
      const files = Array.from(event.target.files || []);
      if (!files.length) {
        setStatus('No document was selected for upload.', 'error');
        render();
        return;
      }

      if (APP_MODE === 'live') {
        try {
          await uploadLiveFiles(files);
        } catch (error) {
          setStatus(error.message, 'error');
          try { await refreshLiveState(); } catch (_) {}
          render();
        }
        event.target.value = '';
        return;
      }

      const file = files[0];
      if (!file.name.toLowerCase().endsWith('.pdf')) {
        setStatus('Unsupported file type. NEXUS only accepts PDF documents.', 'error');
        render();
        return;
      }

      if (file.size <= 0) {
        setStatus('The selected document is empty.', 'error');
        render();
        return;
      }

      const docId = `DOC-${String(S.documents.length + 1).padStart(2, '0')}`;
      const pendingDoc = {
        id: docId,
        name: file.name,
        pages: 12,
        status: 'Parsing',
        concepts: 0,
        relations: 0,
        evidence: 0,
        demo: true
      };

      S.documents = [pendingDoc, ...S.documents];
      S.view = 'sources';
      setStatus(`Parsing ${file.name}...`, 'info');
      render();

      window.setTimeout(() => {
        const processedDoc = S.documents.find((doc) => doc.id === docId);
        if (processedDoc) {
          processedDoc.status = 'Processed';
          processedDoc.concepts = 8;
          processedDoc.relations = 10;
          processedDoc.evidence = 2;
        }

        const demoConcepts = ['Battery Degradation', 'Thermal Stress', 'Thermal Runaway', 'Cycle Life', 'Temperature', 'Electrolyte Stability', 'Charging Rate', 'Lithium-Ion Battery'];
        const hasUploadedConcept = demoConcepts.some((name) => S.selectedConcept === name);
        if (!hasUploadedConcept) {
          S.selectedConcept = 'Battery Degradation';
        }

        setStatus(`${file.name} processed successfully. Graph state refreshed.`, 'success');
        persistStateSnapshot();
        render();
      }, 700);
      event.target.value = '';
    });
  }
}

function startInvestigation() {
  if (!S.documents.length) {
    setStatus('No source documents are available to investigate.', 'error');
    render();
    return;
  }

  if (S.pipelineTimer) {
    clearInterval(S.pipelineTimer);
  }

  const gap = getActiveGap();
  const investigationId = `INV-${String(7 + S.investigations.length).padStart(3, '0')}`;
  S.activeInvestigation = {
    id: investigationId,
    gap: gap.description,
    question: getQuestionForGap(gap),
    status: 'PARTIALLY_SUPPORTED',
    confidence: 68,
    pipeline: ['Gap', 'Question', 'Retrieve', 'Hypothesis', 'Challenge', 'Verify', 'Update']
  };
  S.pipelineStep = 0;
  S.view = 'invest';
  S.knowledgeUpdated = false;
  setStatus('Running autonomous investigation...', 'info');
  render();

  S.pipelineTimer = setInterval(() => {
    if (S.pipelineStep < 7) {
      S.pipelineStep += 1;
      render();
    }
    if (S.pipelineStep >= 7) {
      const completedInvestigation = {
        id: investigationId,
        status: 'Partially supported',
        confidence: 0.68,
        label: gap.description
      };
      const exists = S.investigations.some((entry) => entry.id === investigationId);
      if (!exists) {
        S.investigations.unshift(completedInvestigation);
      }
      setStatus('Investigation completed and verification is ready.', 'success');
      clearInterval(S.pipelineTimer);
      S.pipelineTimer = null;
      persistStateSnapshot();
      render();
    }
  }, 1100);
}

function updateKnowledgeGraph() {
  if (!S.activeInvestigation) {
    setStatus('No active investigation was completed, so the graph could not be updated.', 'error');
    render();
    return;
  }

  S.relationships = [
    ...S.relationships.filter((rel) => !(rel.source === 'Battery Degradation' && rel.target === 'Thermal Runaway')),
    { source: 'Battery Degradation', target: 'Thermal Runaway', type: 'may increase', confidence: 0.68, status: 'weak', claim: 'Battery degradation may increase thermal runaway risk under elevated temperature conditions.', sourceDoc: 'INV-007', page: null }
  ];

  S.gaps = [
    {
      id: 'GAP-008',
      type: 'Missing Relationship',
      description: 'Electrolyte Stability → ??? → Thermal Runaway',
      reason: 'The system has identified a new unresolved safety question after updating the battery degradation pathway.',
      relatedConcepts: ['Electrolyte Stability', 'Thermal Runaway'],
      priority: 'Medium',
      coverage: '36%',
      confidence: 0.69,
      status: 'Open'
    },
    ...S.gaps.filter((gap) => gap.id !== 'GAP-007')
  ];

  S.knowledgeUpdated = true;
  S.nextGapFound = true;
  S.view = 'graph';
  setStatus('Knowledge graph updated. Next gap detected.', 'success');
  persistStateSnapshot();
  render();
}

let LAST_RENDERED_VIEW = null;

function render() {
  updateNav();
  const viewChanged = LAST_RENDERED_VIEW !== S.view;
  LAST_RENDERED_VIEW = S.view;
  const modeBanner = APP_MODE === 'demo'
    ? '<div class="card status-banner status-info mode-banner">DEMO MODE · Seeded sample data and simulated investigation</div>'
    : '';
  $('#m').innerHTML = `<div class="view-content${viewChanged ? ' view-enter' : ''}">${modeBanner}${renderStatusBanner()}${renderCurrentScreen()}</div>`;
  bindEvents();
}

render();
if (APP_MODE === 'live') {
  refreshLiveState().then(render).catch((error) => {
    setStatus(error.message, 'error');
    render();
  });
}

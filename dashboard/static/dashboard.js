/**
 * Smart Home Decision Network — Live Dashboard & 100 Events Text Window
 * LEFT: streaming inference · CENTER: 26-node network stage + Scenario Text Card · RIGHT: MEU decision
 * Modes: "test" (full test-set replay) and "demo" (100 curated distinct events).
 */

// ============================================================================
// State
// ============================================================================
const state = {
    networkData: null,
    nodesById: {},
    selectedNode: null,
    ws: null,
    wsEverConnected: false,
    isPlaying: false,
    currentSpeed: 1.0,
    totalWindows: 0,
    currentIndex: 0,
    currentPosteriors: null,
    lastEvidence: null,
    latencyEma: null,
    // Mode: "test" = full test-set replay   "demo" = 100-window demo dataset
    mode: 'test',
    demoWindows: [],          // cached from /api/demo/all
    demoTotalWindows: 0,
    demoIndex: 0,
    // Text Window Drawer / Modal state
    isTextWindowOpen: false,
    twmFilterAction: 'all',
    twmFilterQuery: '',
    twmSelectedIdx: 0,
};

// ============================================================================
// DOM Elements Map
// ============================================================================
const $ = (id) => document.getElementById(id);
const els = {
    // Header & Status
    connectionStatus:  $('connectionStatus'),
    currentWindow:     $('currentWindow'),
    totalWindows:      $('totalWindows'),
    clockValue:        $('clockValue'),
    btnModeTest:       $('btnModeTest'),
    btnModeDemo:       $('btnModeDemo'),
    btnToggleTextWindow: $('btnToggleTextWindow'),

    // Network Canvas
    networkSvg:        d3.select('#networkSvg'),
    nodeInfo:          $('nodeInfo'),
    networkLegend:     $('networkLegend'),
    pulseStatus:       $('pulseStatus'),

    // Left Panel: Inference
    streamChip:        $('streamChip'),
    evidenceGrid:      $('evidenceGrid'),
    posteriorsGrid:    $('posteriorsGrid'),

    // Right Panel: Decision Engine
    actionName:        $('actionName'),
    actionConfidence:  $('actionConfidence'),
    selectedAction:    $('selectedAction'),
    utilitiesRows:     $('utilitiesRows'),
    statesGrid:        $('statesGrid'),
    comparisonGrid:    $('comparisonGrid'),


    // Transport & Timeline Dock
    btnPlay:           $('btnPlay'),
    btnPause:          $('btnPause'),
    btnPrev:           $('btnPrev'),
    btnNext:           $('btnNext'),
    speedSelect:       $('speedSelect'),
    actionFilterSelect:$('actionFilterSelect'),
    timeline:          $('timeline'),
    timelineProgress:  $('timelineProgress'),
    timelineKnob:      $('timelineKnob'),
    timelineBuffer:    $('timelineBuffer'),
    timelineLabel:     $('timelineLabel'),
    timelineHome:      $('timelineHome'),
    timelineTimestamp: $('timelineTimestamp'),
    windowPulse:       $('windowPulse'),
    latencyDisplay:    $('latencyDisplay'),

    // 100 Events Text Window Modal / Drawer
    textWindowOverlay:     $('textWindowOverlay'),
    btnCloseTextWindow:    $('btnCloseTextWindow'),
    twmSearchInput:        $('twmSearchInput'),
    twmSearchClear:        $('twmSearchClear'),
    twmActionChips:        $('twmActionChips'),
    twmVisibleCount:       $('twmVisibleCount'),
    twmEventList:          $('twmEventList'),
    twmSelectedBadge:      $('twmSelectedBadge'),
    btnLoadSelectedToDash: $('btnLoadSelectedToDash'),
    btnCopyEventText:      $('btnCopyEventText'),
    twmDetailScenario:     $('twmDetailScenario'),
    twmDetailExpl:         $('twmDetailExpl'),
    twmDetailEUs:          $('twmDetailEUs'),
    twmDetailEvidence:     $('twmDetailEvidence'),
    twmDetailPre:          $('twmDetailPre'),
};

// ============================================================================
// Colors & Action Mappings
// ============================================================================
const CATEGORY_COLORS = {
    evidence:   '#6baed6',
    behavioral: '#74c476',
    decision:   '#fd8d3c',
    utility:    '#9e9ac8'
};

const STATE_COLORS = {
    'Occupied':    '#58a6ff', 'Empty':       '#6e7681',
    'Inactive':    '#6e7681', 'Low':         '#58a6ff',
    'Moderate':    '#d29922', 'High':        '#f85149',
    'Awake':       '#d29922', 'Sleeping':    '#a371f7',
    'Yes':         '#f85149', 'No':          '#3fb950',
    'Normal':      '#3fb950', 'Unusual':     '#f85149',
    'None':        '#6e7681', 'Unavailable': '#30363d'
};

const ACTION_COLORS = {
    'No_Action':       '#8b949e',
    'Monitor':         '#58a6ff',
    'Notify_Resident': '#a371f7',
    'Silent_Alert':    '#d29922',
    'Local_Alert':     '#f85149'
};

const ACTION_LABELS = {
    'No_Action':       'No Action',
    'Monitor':         'Monitor',
    'Notify_Resident': 'Notify Resident',
    'Silent_Alert':    'Silent Alert',
    'Local_Alert':     'Local Alert'
};

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
}[c]));

// ============================================================================
// Network visualization (D3)
// ============================================================================
async function loadNetwork() {
    try {
        const res = await fetch('/api/network');
        state.networkData = await res.json();
        state.nodesById = Object.fromEntries(state.networkData.nodes.map(n => [n.id, n]));
        renderNetwork();
        renderLegend();
    } catch (err) {
        console.error('Failed to load network:', err);
    }
}

function renderNetwork() {
    const { nodes, edges } = state.networkData;
    const svg = els.networkSvg;
    svg.selectAll('*').remove();

    const defs = svg.append('defs');
    defs.append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '-0 -5 10 10')
        .attr('refX', 13)
        .attr('refY', 0)
        .attr('orient', 'auto')
        .attr('markerWidth', 7)
        .attr('markerHeight', 7)
        .append('path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('fill', '#30363d');

    const g = svg.append('g');

    const zoom = d3.zoom()
        .scaleExtent([0.3, 3])
        .on('zoom', (event) => g.attr('transform', event.transform));
    svg.call(zoom);
    state.zoom = zoom;

    const width  = svg.node().clientWidth  || 800;
    const height = svg.node().clientHeight || 600;
    const xScale = d3.scaleLinear().domain([0, 1]).range([70, width - 70]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([64, height - 56]);

    if (state.networkData.stages) {
        g.selectAll('.stage-label')
            .data(state.networkData.stages)
            .enter()
            .append('text')
            .attr('class', 'stage-label')
            .attr('x', d => xScale(d.x))
            .attr('y', 26)
            .attr('text-anchor', 'middle')
            .text(d => `${d.label} (${d.count})`);
    }

    g.selectAll('.network-edge')
        .data(edges)
        .enter()
        .append('path')
        .attr('class', 'network-edge')
        .attr('id', d => `edge-${d.source}--${d.target}`)
        .attr('d', d => {
            const s = state.nodesById[d.source], t = state.nodesById[d.target];
            if (!s || !t) return '';
            const sx = xScale(s.x), sy = yScale(s.y);
            const tx = xScale(t.x), ty = yScale(t.y);
            const dx = tx - sx, dy = ty - sy;
            if (Math.abs(dx) < 1) {
                const bulge = Math.max(30, width * 0.06);
                return `M${sx},${sy} C${sx + bulge},${sy + dy * 0.25} ${sx + bulge},${sy + dy * 0.75} ${tx},${ty}`;
            }
            const dr = Math.hypot(dx, dy) * 1.4;
            return `M${sx},${sy}A${dr},${dr} 0 0,1 ${tx},${ty}`;
        })
        .attr('marker-end', 'url(#arrowhead)');

    const node = g.selectAll('.network-node')
        .data(nodes)
        .enter()
        .append('g')
        .attr('class', 'network-node')
        .attr('id', d => `node-${d.id}`)
        .attr('transform', d => `translate(${xScale(d.x)},${yScale(d.y)})`)
        .attr('role', 'button')
        .attr('tabindex', 0)
        .attr('aria-label', d => `${d.label}, ${d.category}`)
        .on('click', (event, d) => selectNode(d))
        .on('mouseover', showNodeTooltip)
        .on('mouseout', hideNodeTooltip)
        .on('keydown', (event, d) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                selectNode(d);
            }
        });

    node.append('circle')
        .attr('class', 'node-circle')
        .attr('r', d => d.category === 'behavioral' ? 22 : 17)
        .attr('fill', d => d.color)
        .attr('fill-opacity', d => d.category === 'behavioral' ? 0.9 : 0.55)
        .attr('stroke', '#30363d')
        .attr('stroke-width', 1.2);

    node.append('circle')
        .attr('class', 'node-ring')
        .attr('r', d => d.category === 'behavioral' ? 27 : 21)
        .attr('fill', 'none')
        .attr('stroke', d => d.color)
        .attr('stroke-width', 1.2)
        .attr('stroke-dasharray', '3,4')
        .attr('opacity', 0.45);

    node.append('text')
        .attr('class', 'node-label')
        .attr('y', d => d.category === 'behavioral' ? 39 : 31)
        .attr('text-anchor', 'middle')
        .each(function(d) {
            const self = d3.select(this);
            const maxWidth = d.category === 'behavioral' ? 140 : 110;
            const text = d.label;
            self.text(text);
            if (this.getComputedTextLength() > maxWidth) {
                let truncated = text;
                while (truncated.length > 3 && this.getComputedTextLength() > maxWidth) {
                    truncated = truncated.slice(0, -1);
                    self.text(truncated + '…');
                }
            }
        });

    fitToView(g, svg, zoom, width, height);
}

function fitToView(g, svg, zoom, width, height) {
    try {
        const bounds = g.node().getBBox();
        if (!bounds.width || !bounds.height) return;
        const scale = Math.min(width / bounds.width, height / bounds.height) * 0.93;
        const tx = (width - bounds.width * scale) / 2 - bounds.x * scale;
        const ty = (height - bounds.height * scale) / 2 - bounds.y * scale;
        svg.call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
    } catch (err) { /* bbox unavailable */ }
}

function renderLegend() {
    const categories = [
        { label: 'Evidence (18)',   color: CATEGORY_COLORS.evidence },
        { label: 'Behavioral (6)', color: CATEGORY_COLORS.behavioral },
        { label: 'Decision (1)',    color: CATEGORY_COLORS.decision },
        { label: 'Utility (1)',     color: CATEGORY_COLORS.utility }
    ];
    els.networkLegend.innerHTML = categories.map(c => `
        <span class="legend-item">
            <span class="legend-color" style="background:${c.color}"></span>${c.label}
        </span>`).join('');
}

function selectNode(node) {
    state.selectedNode = node;
    els.networkSvg.selectAll('.network-node').classed('selected', d => d.id === node.id);

    const categoryLabels = {
        evidence:   'Observable Evidence',
        behavioral: 'Inferred Behavioral',
        decision:   'Decision Action',
        utility:    'Utility Function'
    };

    let posteriorHtml = '';
    const post = state.currentPosteriors?.[node.id];
    if (post) {
        const rows = Object.entries(post).sort((a, b) => b[1] - a[1]).map(([st, p]) =>
            `<div class="detail-row"><span class="detail-label">${esc(st)}</span><span class="detail-value">${(p * 100).toFixed(1)}%</span></div>`
        ).join('');
        posteriorHtml = `<div class="node-posterior">${rows}</div>`;
    }

    els.nodeInfo.innerHTML = `
        <div class="detail">
            <div class="detail-title">${esc(node.label)}</div>
            <div class="detail-row"><span class="detail-label">ID</span><span class="detail-value">${esc(node.id)}</span></div>
            <div class="detail-row"><span class="detail-label">Role</span><span class="detail-value" style="color:${node.color}">${categoryLabels[node.category] || esc(node.category)}</span></div>
        </div>
        ${posteriorHtml}`;
}

let tooltip = null;
function showNodeTooltip(node, event) {
    hideNodeTooltip();
    tooltip = d3.select('body').append('div')
        .style('position', 'absolute')
        .style('padding', '7px 11px')
        .style('background', 'rgba(22,27,34,0.96)')
        .style('border', '1px solid #30363d')
        .style('border-radius', '6px')
        .style('font-size', '0.72rem')
        .style('pointer-events', 'none')
        .style('z-index', 1000)
        .style('left', Math.min(event.pageX + 10, window.innerWidth - 180) + 'px')
        .style('top', (event.pageY - 24) + 'px')
        .html(`<strong>${esc(node.label)}</strong><br><span style="color:#8b949e">${esc(node.id)}</span>`);
}

function hideNodeTooltip() {
    if (tooltip) { tooltip.remove(); tooltip = null; }
}

function animateNodesForWindow(data) {
    const svgNodes = els.networkSvg.selectAll('.network-node');
    svgNodes.classed('active-evidence', false).classed('active-behavioral', false);
    els.networkSvg.selectAll('.network-edge').classed('edge-flow', false);

    const activeEvidence = Object.entries(data.evidence || {})
        .filter(([, v]) => v !== 'Unavailable' && v !== 'None' && v !== 'No')
        .map(([k]) => k);

    activeEvidence.forEach(id => {
        const sel = d3.select(`#node-${CSS.escape(id)}`);
        if (!sel.empty()) sel.classed('active-evidence', true);
    });

    Object.entries(data.posteriors || {}).forEach(([varName]) => {
        const sel = d3.select(`#node-${CSS.escape(varName)}`);
        if (sel.empty()) return;
        sel.classed('active-behavioral', true);
    });

    Object.entries(data.posteriors || {}).forEach(([varName, dist]) => {
        const maxProb = Math.max(...Object.values(dist));
        if (maxProb >= 0.75) {
            state.networkData.edges
                .filter(e => e.target === varName)
                .forEach(e => {
                    const edge = d3.select(`#edge-${CSS.escape(e.source)}--${CSS.escape(e.target)}`);
                    if (!edge.empty()) edge.classed('edge-flow', true);
                });
        }
    });
}

// ============================================================================
// WebSocket
// ============================================================================
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = window.__SMART_HOME_WS_HOST || window.location.host;
    state.ws = new WebSocket(`${protocol}//${wsHost}/ws`);

    state.ws.onopen = () => {
        updateConnectionStatus(true);
        state.wsEverConnected = true;
    };
    state.ws.onmessage = (event) => {
        let msg;
        try { msg = JSON.parse(event.data); } catch { return; }
        handleWebSocketMessage(msg);
    };
    state.ws.onclose = () => {
        updateConnectionStatus(false);
        setTimeout(connectWebSocket, 3000);
    };
    state.ws.onerror = () => { /* onclose follows */ };
}

function updateConnectionStatus(connected) {
    els.connectionStatus.classList.toggle('connected', connected);
    els.connectionStatus.querySelector('span:last-child').textContent =
        connected ? 'Live' : 'Disconnected';
}

function handleWebSocketMessage(msg) {
    if (msg.type === 'update') updateDashboard(msg.data);
    else if (msg.type === 'error') console.warn('Server:', msg.message);
}

function sendWSMessage(msg) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify(msg));
    }
}

// ============================================================================
// Mode toggle (Test ↔ Demo)
// ============================================================================
function setupModeToggle() {
    [els.btnModeTest, els.btnModeDemo].forEach(btn => {
        if (!btn) return;
        btn.addEventListener('click', async () => {
            const newMode = btn.dataset.mode;
            if (newMode === state.mode) return;
            if (state.isPlaying) stopLive();
            setMode(newMode);
        });
    });
}

async function setMode(newMode) {
    state.mode = newMode;
    if (els.btnModeTest) els.btnModeTest.classList.toggle('active', newMode === 'test');
    if (els.btnModeDemo) els.btnModeDemo.classList.toggle('active', newMode === 'demo');

    state.currentIndex = 0;
    state.demoIndex    = 0;

    if (newMode === 'demo') {
        await loadDemoWindows();
        showDemoWindow(0);
    } else {
        await initialLoadTestSet();
    }
}

async function loadDemoWindows() {
    if (state.demoWindows.length > 0) return state.demoWindows;
    try {
        const res = await fetch('/api/demo/all');
        const json = await res.json();
        state.demoWindows = json.windows || [];
        state.demoTotalWindows = json.total || 0;
        console.log(`Loaded ${state.demoTotalWindows} demo windows`);
        return state.demoWindows;
    } catch (err) {
        console.error('Failed to load demo windows:', err);
        return [];
    }
}

function showDemoWindow(idx) {
    if (!state.demoWindows.length) return;
    idx = Math.max(0, Math.min(idx, state.demoWindows.length - 1));
    state.demoIndex = idx;
    const w = state.demoWindows[idx];
    const data = { ...w, index: idx, total_windows: state.demoWindows.length };
    updateDashboard(data);

    // Sync Text Window list selection if open
    if (state.isTextWindowOpen) {
        selectTextWindowEvent(idx, false);
    }
}

// ============================================================================
// Live replay controls
// ============================================================================
function startLive() {
    if (state.isPlaying) return;
    state.isPlaying = true;
    els.btnPlay.disabled  = true;
    els.btnPause.disabled = false;
    els.pulseStatus.textContent = 'LIVE — streaming 5-second windows';
    els.pulseStatus.classList.add('live');
    els.streamChip.textContent = 'streaming';
    els.streamChip.classList.add('live');

    if (state.mode === 'demo') {
        state._demoTimer = setInterval(() => {
            if (!state.isPlaying) return;
            navigateDemo(1);
        }, state.currentSpeed * 1000);
    } else {
        sendWSMessage({ type: 'play', speed: state.currentSpeed, mode: 'test' });
    }
}

function stopLive() {
    state.isPlaying = false;
    els.btnPlay.disabled  = false;
    els.btnPause.disabled = true;
    els.pulseStatus.textContent = 'Paused';
    els.pulseStatus.classList.remove('live');
    els.streamChip.textContent = 'paused';
    els.streamChip.classList.remove('live');

    if (state._demoTimer) { clearInterval(state._demoTimer); state._demoTimer = null; }
    sendWSMessage({ type: 'pause' });
}

function navigateDemo(delta) {
    if (!state.demoWindows.length) return;
    const filter = els.actionFilterSelect ? els.actionFilterSelect.value : 'all';

    if (filter === 'all') {
        const nextIdx = (state.demoIndex + delta + state.demoWindows.length) % state.demoWindows.length;
        showDemoWindow(nextIdx);
        return;
    }

    // Filtered navigation: find matching window indices
    const matchingIndices = state.demoWindows
        .map((w, i) => w.decision?.selected_action === filter ? i : -1)
        .filter(i => i !== -1);

    if (!matchingIndices.length) {
        showDemoWindow((state.demoIndex + delta + state.demoWindows.length) % state.demoWindows.length);
        return;
    }

    const currentPos = matchingIndices.indexOf(state.demoIndex);
    let nextPos;
    if (currentPos === -1) {
        nextPos = delta > 0 ? 0 : matchingIndices.length - 1;
    } else {
        nextPos = (currentPos + delta + matchingIndices.length) % matchingIndices.length;
    }
    showDemoWindow(matchingIndices[nextPos]);
}

function setupControls() {
    els.btnPlay.addEventListener('click', startLive);
    els.btnPause.addEventListener('click', stopLive);

    els.btnPrev.addEventListener('click', () => {
        if (state.mode === 'demo') navigateDemo(-1);
        else stepWindow(-1);
    });
    els.btnNext.addEventListener('click', () => {
        if (state.mode === 'demo') navigateDemo(1);
        else stepWindow(1);
    });

    els.speedSelect.addEventListener('change', (e) => {
        state.currentSpeed = parseFloat(e.target.value);
        if (state.isPlaying && state.mode === 'demo') {
            clearInterval(state._demoTimer);
            state._demoTimer = setInterval(() => {
                if (!state.isPlaying) return;
                navigateDemo(1);
            }, state.currentSpeed * 1000);
        } else {
            sendWSMessage({ type: 'speed', speed: state.currentSpeed });
        }
    });

    // Action filter jump selector in dock
    if (els.actionFilterSelect) {
        els.actionFilterSelect.addEventListener('change', () => {
            const chosen = els.actionFilterSelect.value;
            if (state.mode !== 'demo') {
                setMode('demo');
            }
            if (chosen !== 'all' && state.demoWindows.length) {
                const targetIdx = state.demoWindows.findIndex(w => w.decision?.selected_action === chosen);
                if (targetIdx !== -1) showDemoWindow(targetIdx);
            }
        });
    }

    // Timeline scrubber
    const seek = (event) => {
        const rect = els.timeline.getBoundingClientRect();
        const frac = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
        if (state.mode === 'demo') {
            showDemoWindow(Math.floor(frac * state.demoWindows.length));
        } else {
            jumpToWindow(Math.floor(frac * state.totalWindows));
        }
    };
    let scrubbing = false;
    els.timeline.addEventListener('pointerdown', (e) => { scrubbing = true; els.timeline.setPointerCapture(e.pointerId); seek(e); });
    els.timeline.addEventListener('pointermove', (e) => { if (scrubbing) seek(e); });
    els.timeline.addEventListener('pointerup', () => { scrubbing = false; });
}

async function stepWindow(delta) {
    const next = Math.min(state.totalWindows - 1, Math.max(0, state.currentIndex + delta));
    await jumpToWindow(next);
}

async function jumpToWindow(index) {
    try {
        const res = await fetch(`/api/window/${index}`);
        const data = await res.json();
        if (!data.error) {
            updateDashboard(data);
            sendWSMessage({ type: 'jump', index });
        }
    } catch (err) {
        console.error('Failed to jump:', err);
    }
}

// ============================================================================
// Dashboard updates (per window)
// ============================================================================
function updateDashboard(data) {
    state.currentIndex     = data.index;
    state.currentPosteriors = data.posteriors;

    const renderStart = performance.now();

    const total = data.total_windows || (state.mode === 'demo' ? state.demoTotalWindows : state.totalWindows);
    if (state.mode === 'test') state.totalWindows = total;

    els.currentWindow.textContent      = data.index + 1;
    els.totalWindows.textContent       = total;
    els.timelineLabel.textContent      = `window ${data.index + 1} / ${total}`;
    els.timelineHome.textContent       = data.home_id || '—';
    els.timelineTimestamp.textContent  = data.timestamp || '—';
    els.clockValue.textContent         = (data.timestamp || '').slice(11) || '—';

    renderEvidence(data.evidence);
    renderPosteriors(data.posteriors);
    renderComparison(data.predictions, data.true_labels);
    renderDecision(data.decision);
    animateNodesForWindow(data);

    if (typeof data.index === 'number' && total) {
        const frac = (data.index / total) * 100;
        els.timelineProgress.style.width = frac + '%';
        els.timelineKnob.style.left      = frac + '%';
        els.timelineBuffer.style.width   = '100%';
    }

    els.windowPulse.classList.remove('beat');
    void els.windowPulse.offsetWidth;
    els.windowPulse.classList.add('beat');

    const latency = performance.now() - renderStart;
    state.latencyEma = state.latencyEma === null ? latency : state.latencyEma * 0.8 + latency * 0.2;
    els.latencyDisplay.textContent = `${state.latencyEma.toFixed(1)} ms`;
}

function renderEvidence(evidence) {
    const evidenceNodes = state.networkData?.nodes.filter(n => n.category === 'evidence') || [];
    els.evidenceGrid.innerHTML = evidenceNodes.map(node => {
        const value     = evidence[node.id] || 'Unavailable';
        const isUnavail = value === 'Unavailable';
        const changed   = state.lastEvidence && state.lastEvidence[node.id] !== value;
        return `
            <div class="evidence-item ${isUnavail ? 'unavailable' : ''} ${changed ? 'changed' : ''}">
                <span class="evidence-name" title="${esc(node.id)}">${esc(node.label)}</span>
                <span class="evidence-value" style="color:${getStateColor(value)}">${esc(value)}</span>
            </div>`;
    }).join('');
    state.lastEvidence = { ...evidence };
}

function renderPosteriors(posteriors) {
    const behavioralNodes = state.networkData?.nodes.filter(n => n.category === 'behavioral') || [];
    els.posteriorsGrid.innerHTML = behavioralNodes.map(node => {
        const dist     = posteriors[node.id] || {};
        const maxState = Object.entries(dist).reduce((a, b) => a[1] > b[1] ? a : b, ['', 0])[0];
        const maxProb  = dist[maxState] || 0;
        return `
            <div class="posterior-card">
                <div class="posterior-header">
                    <span class="posterior-name">${esc(node.label)}</span>
                    <span class="posterior-max" style="background:${getStateColor(maxState)}20;color:${getStateColor(maxState)}">${esc(maxState)} ${(maxProb * 100).toFixed(1)}%</span>
                </div>
                <div class="posterior-bars">
                    ${Object.entries(dist).map(([st, p]) => `
                        <div class="posterior-bar">
                            <span class="posterior-bar-label">${esc(st)}</span>
                            <div class="posterior-bar-track">
                                <div class="posterior-bar-fill state-${esc(st.toLowerCase())}" style="width:${(p * 100).toFixed(1)}%"></div>
                            </div>
                            <span class="posterior-bar-value">${(p * 100).toFixed(1)}%</span>
                        </div>`).join('')}
                </div>
            </div>`;
    }).join('');

    if (state.selectedNode && state.currentPosteriors?.[state.selectedNode.id]) {
        const stillThere = document.querySelector('#nodeInfo .detail-title');
        if (stillThere) selectNode(state.selectedNode);
    }
}

function renderComparison(predictions, trueLabels) {
    const behavioralNodes = state.networkData?.nodes.filter(n => n.category === 'behavioral') || [];
    els.comparisonGrid.innerHTML = behavioralNodes.map(node => {
        const pred    = predictions[node.id] || '—';
        const trueVal = trueLabels[node.id]  || '—';
        const match   = pred === trueVal;
        return `
            <div class="comparison-card ${match ? 'correct' : 'incorrect'}">
                <span class="comparison-name" title="${esc(node.id)}">${esc(node.label)}</span>
                <span class="comparison-values">
                    <span class="comparison-value predicted">${esc(pred)}</span>
                    <span class="comparison-value true ${match ? 'match' : 'mismatch'}">${esc(trueVal)}</span>
                </span>
            </div>`;
    }).join('');
}

function renderDecision(decision) {
    if (!decision) return;
    const { selected_action, expected_utilities, state_probabilities } = decision;

    els.actionName.textContent = ACTION_LABELS[selected_action] || selected_action.replace(/_/g, ' ');
    els.actionName.style.color = ACTION_COLORS[selected_action] || 'var(--accent-blue)';
    els.actionConfidence.textContent = `Expected Utility: ${expected_utilities[selected_action].toFixed(2)}`;

    els.selectedAction.className = `selected-action action-${selected_action}`;

    renderUtilityRows(expected_utilities, selected_action);
    renderStatesGrid(state_probabilities);
}

function renderUtilityRows(utilities, selectedAction) {
    const maxEU  = Math.max(...Object.values(utilities));
    const minEU  = Math.min(...Object.values(utilities));
    const range  = maxEU - minEU || 1;

    els.utilitiesRows.innerHTML = Object.entries(utilities).map(([action, eu]) => {
        const isSelected = action === selectedAction;
        const width      = (((eu - minEU) / range) * 100).toFixed(1);
        return `
            <div class="utility-row ${isSelected ? 'selected' : ''}">
                <span class="utility-name">${esc(ACTION_LABELS[action] || action.replace(/_/g, ' '))}</span>
                <div class="utility-track">
                    <div class="utility-fill" style="width:${width}%"></div>
                </div>
                <span class="utility-value">${eu.toFixed(1)}</span>
            </div>`;
    }).join('');
}

function renderStatesGrid(stateProbs) {
    if (!stateProbs) {
        els.statesGrid.innerHTML = '<p class="placeholder">No state probabilities</p>';
        return;
    }
    els.statesGrid.innerHTML = Object.entries(stateProbs).map(([st, p]) => `
        <div class="state-card">
            <div class="state-name">${esc(st.replace(/_/g, ' '))}</div>
            <div class="state-prob" style="color:${getStateColorForProbability(p)}">${(p * 100).toFixed(1)}%</div>
        </div>`).join('');
}

function getStateColor(state)           { return STATE_COLORS[state] || '#8b949e'; }

function getStateColorForProbability(prob) {
    if (prob > 0.7)  return 'var(--accent-green)';
    if (prob > 0.4)  return 'var(--accent-orange)';
    return 'var(--accent-blue)';
}

function maxKey(obj) {
    return Object.entries(obj).reduce((a, b) => a[1] > b[1] ? a : b, ['', -Infinity])[0];
}

// ============================================================================
// 100 Events Text Window Modal & Inspector Implementation
// ============================================================================
function setupTextWindow() {
    if (!els.textWindowOverlay) return;

    // Toggle button in header
    els.btnToggleTextWindow.addEventListener('click', openTextWindow);

    // Expand button on scenario text card
    if (els.stcOpenFullBtn) {
        els.stcOpenFullBtn.addEventListener('click', openTextWindow);
    }

    // Close button & backdrop click
    els.btnCloseTextWindow.addEventListener('click', closeTextWindow);
    els.textWindowOverlay.addEventListener('click', (e) => {
        if (e.target === els.textWindowOverlay) closeTextWindow();
    });

    // Keyboard shortcuts: 'T' toggles, Escape closes
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && state.isTextWindowOpen) {
            closeTextWindow();
        } else if ((e.key === 't' || e.key === 'T') && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
            if (state.isTextWindowOpen) closeTextWindow();
            else openTextWindow();
        }
    });

    // Search input
    els.twmSearchInput.addEventListener('input', (e) => {
        state.twmFilterQuery = e.target.value.toLowerCase().trim();
        const wrap = els.twmSearchInput.closest('.twm-search-wrap');
        if (wrap) wrap.classList.toggle('has-query', Boolean(state.twmFilterQuery));
        renderTextWindowList();
    });

    // Search clear
    if (els.twmSearchClear) {
        els.twmSearchClear.addEventListener('click', () => {
            els.twmSearchInput.value = '';
            state.twmFilterQuery = '';
            els.twmSearchInput.closest('.twm-search-wrap')?.classList.remove('has-query');
            renderTextWindowList();
        });
    }

    // Action filter chips
    els.twmActionChips.querySelectorAll('.twm-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            els.twmActionChips.querySelectorAll('.twm-chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            state.twmFilterAction = chip.dataset.action;
            renderTextWindowList();
        });
    });

    // Load selected event into live dashboard button
    if (els.btnLoadSelectedToDash) {
        els.btnLoadSelectedToDash.addEventListener('click', () => {
            selectTextWindowEvent(state.twmSelectedIdx, true);
            closeTextWindow();
        });
    }

    // Copy event text button
    if (els.btnCopyEventText) {
        els.btnCopyEventText.addEventListener('click', () => {
            const text = els.twmDetailPre.textContent;
            navigator.clipboard.writeText(text).then(() => {
                const old = els.btnCopyEventText.textContent;
                els.btnCopyEventText.textContent = '✓ Copied!';
                setTimeout(() => { els.btnCopyEventText.textContent = old; }, 2000);
            });
        });
    }
}

async function openTextWindow() {
    state.isTextWindowOpen = true;
    els.textWindowOverlay.classList.add('open');
    els.textWindowOverlay.setAttribute('aria-hidden', 'false');

    await loadDemoWindows();
    renderTextWindowList();
    selectTextWindowEvent(state.mode === 'demo' ? state.demoIndex : 0, false);
}

function closeTextWindow() {
    state.isTextWindowOpen = false;
    els.textWindowOverlay.classList.remove('open');
    els.textWindowOverlay.setAttribute('aria-hidden', 'true');
}

function renderTextWindowList() {
    if (!els.twmEventList) return;
    const windows = state.demoWindows;
    if (!windows.length) {
        els.twmEventList.innerHTML = '<p class="placeholder" style="padding:20px;">Loading 100 demo events…</p>';
        return;
    }

    // Filter by action and search query
    const filtered = windows.filter(w => {
        const action = w.decision?.selected_action || '';
        if (state.twmFilterAction !== 'all' && action !== state.twmFilterAction) {
            return false;
        }
        if (state.twmFilterQuery) {
            const q = state.twmFilterQuery;
            const title = (w.scenario_title || '').toLowerCase();
            const desc = (w.description || '').toLowerCase();
            const home = (w.home_id || '').toLowerCase();
            const act = action.toLowerCase();
            const sensors = Object.entries(w.evidence || {}).map(([k, v]) => `${k}:${v}`).join(' ').toLowerCase();
            return title.includes(q) || desc.includes(q) || home.includes(q) || act.includes(q) || sensors.includes(q);
        }
        return true;
    });

    els.twmVisibleCount.textContent = `Showing ${filtered.length} of ${windows.length} events`;

    if (!filtered.length) {
        els.twmEventList.innerHTML = '<p class="placeholder" style="padding:24px;text-align:center;">No matching events found. Try adjusting filters or search.</p>';
        return;
    }

    els.twmEventList.innerHTML = filtered.map(w => {
        const idx = w.window_id ?? w.index ?? 0;
        const action = w.decision?.selected_action || 'No_Action';
        const actionLabel = ACTION_LABELS[action] || action.replace(/_/g, ' ');
        const eu = w.decision?.expected_utilities?.[action];
        const euStr = (typeof eu === 'number') ? `EU: ${eu >= 0 ? '+' : ''}${eu.toFixed(1)}` : '';
        const isSelected = idx === state.twmSelectedIdx;

        // Extract active sensor summary
        const activeSensors = Object.entries(w.evidence || {})
            .filter(([, v]) => v !== 'None' && v !== 'No' && v !== 'Unavailable')
            .map(([k, v]) => `${k.replace('_Activity', '')}:${v}`)
            .slice(0, 4)
            .join(' · ');

        return `
            <div class="twm-card action-border-${action} ${isSelected ? 'active' : ''}" data-idx="${idx}">
                <div class="twm-card-top">
                    <div class="twm-card-tags">
                        <span class="twm-card-id">#${idx + 1}</span>
                        <span class="twm-card-badge action-${action}">${actionLabel}</span>
                        <span class="twm-card-home">${esc(w.home_id || 'Aruba')}</span>
                    </div>
                    <span class="twm-card-eu" style="color:${ACTION_COLORS[action]}">${euStr}</span>
                </div>
                <div class="twm-card-title">${esc(w.scenario_title || `Event #${idx + 1}`)}</div>
                <div class="twm-card-desc">${esc(w.description || '')}</div>
                <div class="twm-card-footer">
                    <span>${esc((w.timestamp || '').slice(11))} · ${esc(w.evidence?.Time_Of_Day || '')}</span>
                    <span class="twm-card-sensors" title="${esc(activeSensors)}">${esc(activeSensors)}</span>
                </div>
            </div>`;
    }).join('');

    // Attach card click handlers
    els.twmEventList.querySelectorAll('.twm-card').forEach(card => {
        card.addEventListener('click', () => {
            const idx = parseInt(card.dataset.idx, 10);
            selectTextWindowEvent(idx, false);
        });
        card.addEventListener('dblclick', () => {
            const idx = parseInt(card.dataset.idx, 10);
            selectTextWindowEvent(idx, true);
            closeTextWindow();
        });
    });
}

function selectTextWindowEvent(idx, loadToDash = false) {
    state.twmSelectedIdx = idx;

    // Update active highlight on cards
    if (els.twmEventList) {
        els.twmEventList.querySelectorAll('.twm-card').forEach(c => {
            c.classList.toggle('active', parseInt(c.dataset.idx, 10) === idx);
        });
    }

    const w = state.demoWindows[idx];
    if (!w) return;

    const action = w.decision?.selected_action || 'No_Action';
    const actionLabel = ACTION_LABELS[action] || action.replace(/_/g, ' ');

    els.twmSelectedBadge.textContent = `Event #${idx + 1} · ${actionLabel}`;
    els.twmSelectedBadge.className = `twm-detail-selected-badge action-${action}`;

    // Scenario & Description
    els.twmDetailScenario.innerHTML = `
        <div style="font-weight:600;margin-bottom:4px;font-size:0.86rem;color:var(--text-primary);">
            ${esc(w.scenario_title || `Event #${idx + 1}`)}
        </div>
        <div style="color:var(--text-secondary);margin-bottom:6px;">
            ${esc(w.description || '')}
        </div>
        <div style="font-size:0.7rem;color:var(--text-muted);font-family:'JetBrains Mono',monospace;">
            Home: <strong>${esc(w.home_id || 'Aruba')}</strong> · Timestamp: <strong>${esc(w.timestamp || '')}</strong> · Time: <strong>${esc(w.evidence?.Time_Of_Day || '')}</strong>
        </div>`;

    // Decision Reasoning
    els.twmDetailExpl.innerHTML = `
        <div style="font-weight:600;margin-bottom:4px;color:${ACTION_COLORS[action]}">
            Optimal Action: ${actionLabel} (Expected Utility: ${w.decision?.expected_utilities?.[action]?.toFixed(2)})
        </div>
        <div>${esc(w.explanation || '')}</div>`;

    // Expected Utilities Table
    if (w.decision?.expected_utilities) {
        const eus = w.decision.expected_utilities;
        const maxEU = Math.max(...Object.values(eus));
        const minEU = Math.min(...Object.values(eus));
        const range = maxEU - minEU || 1;

        const rows = Object.entries(eus).map(([act, val]) => {
            const isSel = act === action;
            const pct = Math.max(5, Math.min(100, ((val - minEU) / range) * 100));
            return `
                <tr class="${isSel ? 'selected' : ''}">
                    <td><strong>${esc(ACTION_LABELS[act] || act)}</strong></td>
                    <td style="font-family:'JetBrains Mono',monospace;text-align:right;">${val >= 0 ? '+' : ''}${val.toFixed(2)}</td>
                    <td style="width:100px;">
                        <div class="twm-eu-bar-wrap">
                            <div class="twm-eu-bar-fill" style="width:${pct}%;background:${ACTION_COLORS[act] || 'var(--accent-blue)'}"></div>
                        </div>
                    </td>
                </tr>`;
        }).join('');

        els.twmDetailEUs.innerHTML = `
            <table class="twm-eu-table">
                <thead><tr><th>Action</th><th style="text-align:right;">Expected Utility</th><th>Relative Rank</th></tr></thead>
                <tbody>${rows}</tbody>
            </table>`;
    }

    // Observed Evidence
    if (w.evidence) {
        const evidenceBadges = Object.entries(w.evidence).map(([k, v]) => {
            const isUnavail = v === 'Unavailable' || v === 'None' || v === 'No';
            return `
                <span class="evidence-badge ${isUnavail ? 'muted' : 'active'}" style="display:inline-block;padding:2px 7px;margin:2px;font-size:0.68rem;border-radius:4px;background:var(--bg-tertiary);border:1px solid var(--border-color);font-family:'JetBrains Mono',monospace;">
                    <span style="color:var(--text-muted);">${esc(k.replace('_Activity', ''))}:</span>
                    <strong style="color:${getStateColor(v)}">${esc(v)}</strong>
                </span>`;
        }).join('');
        els.twmDetailEvidence.innerHTML = `<div style="display:flex;flex-wrap:wrap;gap:4px;">${evidenceBadges}</div>`;
    }

    // Raw JSON representation
    els.twmDetailPre.textContent = JSON.stringify(w, null, 2);

    // If loadToDash is requested, trigger dashboard view
    if (loadToDash) {
        if (state.mode !== 'demo') {
            setMode('demo').then(() => showDemoWindow(idx));
        } else {
            showDemoWindow(idx);
        }
    }
}

// ============================================================================
// Initial load + init
// ============================================================================
async function initialLoadTestSet() {
    try {
        const statsRes = await fetch('/api/stats');
        const stats    = await statsRes.json();
        state.totalWindows = stats.total_windows || 0;
        els.totalWindows.textContent = state.totalWindows;

        const res  = await fetch('/api/window/0');
        const data = await res.json();
        if (!data.error) updateDashboard(data);
    } catch (err) {
        console.error('Initial load failed:', err);
    }
}

async function init() {
    await loadNetwork();
    setupControls();
    setupModeToggle();
    setupTextWindow();
    connectWebSocket();

    // Pre-cache 100 demo windows in background
    await loadDemoWindows();

    // Initial test set load
    await initialLoadTestSet();
}

document.addEventListener('DOMContentLoaded', init);

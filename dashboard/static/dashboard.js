/**
 * Smart Home Decision Network — Live Dashboard
 * LEFT: streaming inference · CENTER: 26-node network stage · RIGHT: MEU decision
 * Live replay: auto-plays windows over WebSocket, 1 update = 1 five-second sensor window.
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
    latencyEma: null
};

// ============================================================================
// DOM
// ============================================================================
const $ = (id) => document.getElementById(id);
const els = {
    connectionStatus: $('connectionStatus'),
    currentWindow: $('currentWindow'),
    totalWindows: $('totalWindows'),
    networkSvg: d3.select('#networkSvg'),
    nodeInfo: $('nodeInfo'),
    networkLegend: $('networkLegend'),
    evidenceGrid: $('evidenceGrid'),
    posteriorsGrid: $('posteriorsGrid'),
    comparisonGrid: $('comparisonGrid'),
    actionName: $('actionName'),
    actionConfidence: $('actionConfidence'),
    utilitiesRows: $('utilitiesRows'),
    statesGrid: $('statesGrid'),
    latencyDisplay: $('latencyDisplay'),
    btnPlay: $('btnPlay'),
    btnPause: $('btnPause'),
    btnPrev: $('btnPrev'),
    btnNext: $('btnNext'),
    speedSelect: $('speedSelect'),
    pulseStatus: $('pulseStatus'),
    streamChip: $('streamChip'),
    clockValue: $('clockValue'),
    timeline: $('timeline'),
    timelineProgress: $('timelineProgress'),
    timelineKnob: $('timelineKnob'),
    timelineBuffer: $('timelineBuffer'),
    timelineLabel: $('timelineLabel'),
    timelineHome: $('timelineHome'),
    timelineTimestamp: $('timelineTimestamp'),
    windowPulse: $('windowPulse')
};

// ============================================================================
// Colors
// ============================================================================
const CATEGORY_COLORS = {
    evidence: '#6baed6',
    behavioral: '#74c476',
    decision: '#fd8d3c',
    utility: '#9e9ac8'
};

const STATE_COLORS = {
    'Occupied': '#58a6ff', 'Empty': '#6e7681',
    'Inactive': '#6e7681', 'Low': '#58a6ff', 'Moderate': '#d29922', 'High': '#f85149',
    'Awake': '#d29922', 'Sleeping': '#a371f7',
    'Yes': '#f85149', 'No': '#3fb950',
    'Normal': '#3fb950', 'Unusual': '#f85149',
    'None': '#6e7681', 'Unavailable': '#30363d'
};

const ACTION_COLORS = {
    'No_Action': '#6e7681',
    'Monitor': '#58a6ff',
    'Notify_Resident': '#a371f7',
    'Silent_Alert': '#d29922',
    'Local_Alert': '#f85149'
};

const esc = (s) => String(s).replace(/[&<>"']/g, c => ({
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

    const width = svg.node().clientWidth || 800;
    const height = svg.node().clientHeight || 600;
    const xScale = d3.scaleLinear().domain([0, 1]).range([70, width - 70]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([64, height - 56]);

    // Stage captions above each pipeline column (evidence → behavioral → …)
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
                // Same stage column: arc gently to the right using relative bulge
                // Scale bulge based on container width for responsiveness
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
            // Truncate with ellipsis if too wide
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
        { label: 'Evidence (18)', color: CATEGORY_COLORS.evidence },
        { label: 'Behavioral (6)', color: CATEGORY_COLORS.behavioral },
        { label: 'Decision (1)', color: CATEGORY_COLORS.decision },
        { label: 'Utility (1)', color: CATEGORY_COLORS.utility }
    ];
    els.networkLegend.innerHTML = categories.map(c => `
        <span class="legend-item">
            <span class="legend-color" style="background:${c.color}"></span>${c.label}
        </span>`).join('');
}

function selectNode(node) {
    state.selectedNode = node;

    els.networkSvg.selectAll('.network-node')
        .classed('selected', d => d.id === node.id);

    const categoryLabels = {
        evidence: 'Observable Evidence',
        behavioral: 'Inferred Behavioral',
        decision: 'Decision Action',
        utility: 'Utility Function'
    };

    // If a posterior exists for this node, show its current distribution
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

// Node tooltips
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

// Per-window graph animation: highlight evidence + behavioral nodes
function animateNodesForWindow(data) {
    const svgNodes = els.networkSvg.selectAll('.network-node');
    svgNodes.classed('active-evidence', false)
            .classed('active-behavioral', false);
    els.networkSvg.selectAll('.network-edge').classed('edge-flow', false);

    // Evidence nodes that actually observed something this window
    const activeEvidence = Object.entries(data.evidence || {})
        .filter(([, v]) => v !== 'Unavailable' && v !== 'None' && v !== 'No')
        .map(([k]) => k);

    activeEvidence.forEach(id => {
        const sel = d3.select(`#node-${CSS.escape(id)}`);
        if (!sel.empty()) sel.classed('active-evidence', true);
    });

    // Behavioral nodes glow; their strongest state drives node brightness
    Object.entries(data.posteriors || {}).forEach(([varName, dist]) => {
        const sel = d3.select(`#node-${CSS.escape(varName)}`);
        if (sel.empty()) return;
        sel.classed('active-behavioral', true);
    });

    // Light up edges flowing into high-confidence behavioral conclusions
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
    // Support reverse proxies: use explicit backend host if configured, else fall back to window.location.host
    const wsHost = window.__SMART_HOME_WS_HOST || window.location.host;
    state.ws = new WebSocket(`${protocol}//${wsHost}/ws`);

    state.ws.onopen = () => {
        updateConnectionStatus(true);
        if (!state.wsEverConnected) {
            state.wsEverConnected = true;
        }
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
    switch (msg.type) {
        case 'update':
            updateDashboard(msg.data);
            break;
        case 'error':
            console.warn('Server:', msg.message);
            break;
    }
}

function sendWSMessage(msg) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify(msg));
    }
}

// ============================================================================
// Live replay controls
// ============================================================================
function startLive() {
    if (state.isPlaying) return;
    state.isPlaying = true;
    els.btnPlay.disabled = true;
    els.btnPause.disabled = false;
    els.pulseStatus.textContent = 'LIVE — streaming 5-second windows';
    els.pulseStatus.classList.add('live');
    els.streamChip.textContent = 'streaming';
    els.streamChip.classList.add('live');
    sendWSMessage({ type: 'play', speed: state.currentSpeed });
}

function stopLive() {
    state.isPlaying = false;
    els.btnPlay.disabled = false;
    els.btnPause.disabled = true;
    els.pulseStatus.textContent = 'Paused';
    els.pulseStatus.classList.remove('live');
    els.streamChip.textContent = 'paused';
    els.streamChip.classList.remove('live');
    sendWSMessage({ type: 'pause' });
}

function setupControls() {
    els.btnPlay.addEventListener('click', startLive);
    els.btnPause.addEventListener('click', stopLive);
    els.btnPrev.addEventListener('click', () => stepWindow(-1));
    els.btnNext.addEventListener('click', () => stepWindow(1));
    els.speedSelect.addEventListener('change', (e) => {
        state.currentSpeed = parseFloat(e.target.value);
        sendWSMessage({ type: 'speed', speed: state.currentSpeed });
    });

    // Click / drag on the timeline = jump
    const seek = (event) => {
        const rect = els.timeline.getBoundingClientRect();
        const frac = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
        jumpToWindow(Math.floor(frac * state.totalWindows));
    };
    let scrubbing = false;
    els.timeline.addEventListener('pointerdown', (e) => { scrubbing = true; els.timeline.setPointerCapture(e.pointerId); seek(e); });
    els.timeline.addEventListener('pointermove', (e) => { if (scrubbing) seek(e); });
    els.timeline.addEventListener('pointerup', () => { scrubbing = false; });
}

async function stepWindow(delta) {
    const cur = state.currentIndex ?? 0;
    const next = Math.min(state.totalWindows - 1, Math.max(0, cur + delta));
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
    state.currentIndex = data.index;
    state.currentPosteriors = data.posteriors;

    const renderStart = performance.now();

    els.currentWindow.textContent = data.index + 1;
    els.timelineLabel.textContent = `window ${data.index + 1} / ${data.total_windows}`;
    els.timelineHome.textContent = data.home_id;
    els.timelineTimestamp.textContent = data.timestamp;
    els.clockValue.textContent = data.timestamp.slice(11); // HH:MM:SS part

    renderEvidence(data.evidence);
    renderPosteriors(data.posteriors);
    renderComparison(data.predictions, data.true_labels);
    renderDecision(data.decision);
    animateNodesForWindow(data);

    if (typeof data.index === 'number' && data.total_windows) {
        const frac = (data.index / data.total_windows) * 100;
        els.timelineProgress.style.width = frac + '%';
        els.timelineKnob.style.left = frac + '%';
        els.timelineBuffer.style.width = '100%';
    }

    // 5-second window pulse
    els.windowPulse.classList.remove('beat');
    void els.windowPulse.offsetWidth;          // restart animation
    els.windowPulse.classList.add('beat');

    // Latency (EMA-smoothed)
    const latency = performance.now() - renderStart;
    state.latencyEma = state.latencyEma === null ? latency : state.latencyEma * 0.8 + latency * 0.2;
    els.latencyDisplay.textContent = `Latency: ${state.latencyEma.toFixed(1)} ms`;
}

function renderEvidence(evidence) {
    const evidenceNodes = state.networkData?.nodes.filter(n => n.category === 'evidence') || [];
    els.evidenceGrid.innerHTML = evidenceNodes.map(node => {
        const value = evidence[node.id] || 'Unavailable';
        const isUnavail = value === 'Unavailable';
        const changed = state.lastEvidence && state.lastEvidence[node.id] !== value;
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
        const dist = posteriors[node.id] || {};
        const maxState = Object.entries(dist).reduce((a, b) => a[1] > b[1] ? a : b, ['', 0])[0];
        const maxProb = dist[maxState] || 0;
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

    // Keep an open node's posterior fresh
    if (state.selectedNode && state.currentPosteriors?.[state.selectedNode.id]) {
        const stillThere = document.querySelector('#nodeInfo .detail-title');
        if (stillThere) selectNode(state.selectedNode);
    }
}

function renderComparison(predictions, trueLabels) {
    const behavioralNodes = state.networkData?.nodes.filter(n => n.category === 'behavioral') || [];
    els.comparisonGrid.innerHTML = behavioralNodes.map(node => {
        const pred = predictions[node.id] || '—';
        const trueVal = trueLabels[node.id] || '—';
        const match = pred === trueVal;
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

    els.actionName.textContent = selected_action.replace(/_/g, ' ');
    els.actionName.style.color = ACTION_COLORS[selected_action] || 'var(--accent-blue)';
    els.actionConfidence.textContent = `Expected Utility: ${expected_utilities[selected_action].toFixed(2)}`;

    renderUtilityRows(expected_utilities, selected_action);
    renderStatesGrid(state_probabilities);
}

function renderUtilityRows(utilities, selectedAction) {
    const maxEU = Math.max(...Object.values(utilities));
    const minEU = Math.min(...Object.values(utilities));
    const range = maxEU - minEU || 1;

    els.utilitiesRows.innerHTML = Object.entries(utilities).map(([action, eu]) => {
        const isSelected = action === selectedAction;
        const width = (((eu - minEU) / range) * 100).toFixed(1);
        return `
            <div class="utility-row ${isSelected ? 'selected' : ''}">
                <span class="utility-name">${esc(action.replace(/_/g, ' '))}</span>
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

function getStateColor(state) { return STATE_COLORS[state] || '#8b949e'; }

function getStateColorForProbability(prob) {
    if (prob > 0.7) return 'var(--accent-green)';
    if (prob > 0.4) return 'var(--accent-orange)';
    return 'var(--accent-blue)';
}

// ============================================================================
// Initial load + init
// ============================================================================
async function initialLoad() {
    try {
        const statsRes = await fetch('/api/stats');
        const stats = await statsRes.json();
        state.totalWindows = stats.total_windows || 0;
        els.totalWindows.textContent = state.totalWindows;

        const res = await fetch('/api/window/0');
        const data = await res.json();
        if (!data.error) updateDashboard(data);
    } catch (err) {
        console.error('Initial load failed:', err);
    }
}

async function init() {
    await loadNetwork();
    setupControls();
    connectWebSocket();
    await initialLoad();
}

document.addEventListener('DOMContentLoaded', init);

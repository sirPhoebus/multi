let network = null;
let currentAgent = null;
let lastData = null;

// Configuration for Vis.js
const options = {
    nodes: {
        shape: 'dot',
        size: 30,
        font: {
            size: 14,
            color: '#ffffff',
            face: 'Inter'
        },
        borderWidth: 2,
        color: {
            background: '#1e293b',
            border: '#3b82f6',
            highlight: {
                background: '#3b82f6',
                border: '#ffffff'
            }
        }
    },
    edges: {
        width: 2,
        smooth: {
            type: 'continuous'
        },
        arrows: {
            to: { enabled: true, scaleFactor: 1.2 }
        }
    },
    physics: {
        stabilization: false,
        barnesHut: {
            gravitationalConstant: -8000,
            springConstant: 0.04,
            springLength: 95
        }
    },
    layout: {
        randomSeed: 2
    }
};

async function fetchData() {
    try {
        // Cache bust to ensure fresh data
        const response = await fetch('data.json?t=' + new Date().getTime());
        if (!response.ok) throw new Error("No data file");
        const data = await response.json();
        return data;
    } catch (e) {
        console.log("Waiting for simulation data...", e);
        return null;
    }
}

function updateUI(data) {
    if (!data) return;

    // Update header
    document.getElementById('step-counter').innerText = `Step: ${data.step}`;
    document.getElementById('status').innerText = "Live";
    document.getElementById('status').style.color = "#22c55e";

    // Populate Sidebar if needed
    const list = document.getElementById('agent-list');
    const agentIds = Object.keys(data.agents);

    // Initial select
    if (!currentAgent && agentIds.length > 0) {
        currentAgent = agentIds[0];
    }

    list.innerHTML = '';
    agentIds.forEach(id => {
        const info = data.agents[id];
        const el = document.createElement('div');
        el.className = `agent-card ${id === currentAgent ? 'active' : ''}`;
        el.onclick = () => {
            currentAgent = id;
            updateGraph(data);
        };
        el.innerHTML = `
            <span class="agent-name">Agent ${id}</span>
            <span class="agent-score">Best Reward: ${info.best_score.toFixed(1)}</span>
        `;
        list.appendChild(el);
    });
}

function updateGraph(data) {
    if (!currentAgent || !data.agents[currentAgent]) return;

    const agentData = data.agents[currentAgent];
    const container = document.getElementById('mynetwork');

    const nodes = new vis.DataSet(agentData.nodes);
    const edges = new vis.DataSet(agentData.edges);

    const graphData = { nodes, edges };

    if (!network) {
        network = new vis.Network(container, graphData, options);
    } else {
        network.setData(graphData);
    }
}

async function loop() {
    const data = await fetchData();
    if (data) {
        lastData = data;
        updateUI(data);
        updateGraph(data);
    }
    setTimeout(loop, 2000); // Poll every 2s
}

// Start
document.addEventListener('DOMContentLoaded', loop);

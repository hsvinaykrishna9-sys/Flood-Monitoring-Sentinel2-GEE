// Vanilla JS, no build step. Leaflet is the one external library this page
// loads (via CDN, in index.html) - unavoidable for a real interactive map;
// everything else here calls only the FastAPI backend at relative URLs
// (same origin, works identically on localhost and after deployment).

const statusBox = document.getElementById("statusBox");
const content = document.getElementById("content");
const searchForm = document.getElementById("searchForm");
const searchInput = document.getElementById("searchInput");
const suggestionsEl = document.getElementById("suggestions");
const locateBtn = document.getElementById("locateBtn");

let leafletMap = null;
let activeSuggestionIndex = -1;
let currentSuggestions = [];

function iconFor(condition) {
  const c = condition.toLowerCase();
  if (c.includes("thunder")) return "⛈️";
  if (c.includes("snow")) return "❄️";
  if (c.includes("rain") || c.includes("drizzle") || c.includes("shower")) return "\u{1F327}️";
  if (c.includes("fog")) return "\u{1F32B}️";
  if (c.includes("overcast")) return "☁️";
  if (c.includes("partly") || c.includes("cloud")) return "⛅";
  if (c.includes("clear")) return "☀️";
  return "\u{1F324}️";
}

function setStatus(text, isError) {
  statusBox.hidden = false;
  content.hidden = true;
  statusBox.textContent = text;
  statusBox.className = "status-box" + (isError ? " error-text" : "");
}

function fmtHour(iso) { return iso.slice(11, 16); }
function fmtClock(iso) {
  if (!iso) return "--:--";
  const [, time] = iso.split("T");
  return time || "--:--";
}
function fmtDay(iso, index) {
  if (index === 0) return "Today";
  const d = new Date(iso + "T00:00:00");
  return d.toLocaleDateString(undefined, { weekday: "short" });
}

// ---------- UV / AQI helpers ----------

function uvCategory(uv) {
  if (uv == null) return { text: "—", color: "var(--low)", pct: 0 };
  if (uv < 3) return { text: "Low", color: "var(--low)", pct: (uv / 12) * 100 };
  if (uv < 6) return { text: "Moderate", color: "var(--moderate)", pct: (uv / 12) * 100 };
  if (uv < 8) return { text: "High", color: "#ff9a3d", pct: (uv / 12) * 100 };
  if (uv < 11) return { text: "Very High", color: "var(--high)", pct: (uv / 12) * 100 };
  return { text: "Extreme", color: "#c04dff", pct: 100 };
}

function aqiColor(category) {
  if (category === "Good" || category === "Moderate") return "var(--low)";
  if (category === "Unhealthy for Sensitive Groups") return "var(--moderate)";
  if (category === "Unhealthy") return "#ff9a3d";
  return "var(--high)";
}

// ---------- Map ----------

function initMap(lat, lon, label) {
  const mapEl = document.getElementById("map");
  try {
    if (leafletMap) { leafletMap.remove(); leafletMap = null; }
    leafletMap = L.map(mapEl, { zoomControl: true }).setView([lat, lon], 7);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 12,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(leafletMap);

    L.marker([lat, lon]).addTo(leafletMap).bindPopup(label);

    // Live precipitation radar overlay (RainViewer, free public API) -
    // optional: if it fails to load, the base map above still works fine.
    fetch("https://api.rainviewer.com/public/weather-maps.json")
      .then((r) => r.json())
      .then((data) => {
        const frames = (data.radar && data.radar.past) || [];
        if (!frames.length) return;
        const latest = frames[frames.length - 1];
        const tileUrl = `${data.host}${latest.path}/256/{z}/{x}/{y}/2/1_1.png`;
        L.tileLayer(tileUrl, { opacity: 0.55, maxZoom: 12 }).addTo(leafletMap);
      })
      .catch(() => {});
  } catch (err) {
    mapEl.innerHTML = '<div style="padding:16px;color:rgba(255,255,255,0.6);font-size:13px;">Map unavailable right now.</div>';
  }
}

// ---------- Render ----------

function render(data) {
  statusBox.hidden = true;
  content.hidden = false;

  const loc = data.location;
  const nameParts = [loc.name];
  if (loc.admin1) nameParts.push(loc.admin1);
  document.getElementById("locationName").textContent = nameParts.join(", ");

  const cur = data.current;
  document.getElementById("currentTemp").textContent = Math.round(cur.temperature_C) + "°";
  document.getElementById("condition").textContent = cur.condition;
  document.getElementById("feelsLike").textContent =
    cur.feels_like_C != null ? `Feels like ${Math.round(cur.feels_like_C)}°` : "";

  const today = data.daily_outlook[0];
  if (today) {
    document.getElementById("hilo").textContent =
      `H:${Math.round(today.temp_max_C)}°  L:${Math.round(today.temp_min_C)}°`;
  }

  // Details card
  document.getElementById("humidity").textContent = Math.round(cur.humidity_pct) + "%";
  document.getElementById("dewPoint").textContent =
    cur.dew_point_C != null ? Math.round(cur.dew_point_C) + "°" : "—";
  document.getElementById("pressure").textContent =
    cur.pressure_hpa != null ? Math.round(cur.pressure_hpa) + " hPa" : "—";
  document.getElementById("visibility").textContent =
    cur.visibility_km != null ? cur.visibility_km.toFixed(1) + " km" : "—";

  // Wind
  document.getElementById("windSpeed").textContent = Math.round(cur.wind_speed_kmh) + " km/h";

  // Precipitation card
  document.getElementById("precipToday").textContent = (today ? today.rain_mm.toFixed(1) : "0.0") + " mm";
  const nextRainDay = data.daily_outlook.find((d) => d.rain_mm > 0);
  document.getElementById("precipNext").textContent = nextRainDay
    ? `Next rain expected ${fmtDay(nextRainDay.date, data.daily_outlook.indexOf(nextRainDay))}: ${nextRainDay.rain_mm.toFixed(1)} mm`
    : "No rain expected in the next week";

  // UV
  const uv = uvCategory(cur.uv_index);
  document.getElementById("uvValue").textContent = cur.uv_index != null ? cur.uv_index : "—";
  document.getElementById("uvCategory").textContent = uv.text;
  const uvBar = document.getElementById("uvBar");
  uvBar.style.width = Math.min(uv.pct, 100) + "%";
  uvBar.style.background = uv.color;

  // Air quality
  const aqiCard = document.getElementById("aqiBox");
  if (data.air_quality) {
    const aqi = data.air_quality;
    document.getElementById("aqiValue").textContent = Math.round(aqi.us_aqi);
    document.getElementById("aqiCategory").textContent = aqi.category;
    const aqiBar = document.getElementById("aqiBar");
    aqiBar.style.width = Math.min((aqi.us_aqi / 300) * 100, 100) + "%";
    aqiBar.style.background = aqiColor(aqi.category);
    aqiCard.hidden = false;
  } else {
    document.getElementById("aqiValue").textContent = "—";
    document.getElementById("aqiCategory").textContent = "Unavailable for this location";
    document.getElementById("aqiBar").style.width = "0%";
  }

  // Sun & moon
  document.getElementById("sunrise").textContent = fmtClock(data.sun.sunrise);
  document.getElementById("sunset").textContent = fmtClock(data.sun.sunset);
  document.getElementById("moonPhase").textContent = data.moon.name;
  document.getElementById("moonIllum").textContent = data.moon.illumination_pct + "%";

  // Hourly strip
  const hourlyStrip = document.getElementById("hourlyStrip");
  hourlyStrip.innerHTML = "";
  data.hourly_outlook.forEach((h, i) => {
    const el = document.createElement("div");
    el.className = "hour-item";
    el.innerHTML = `
      <div class="h-time">${i === 0 ? "Now" : fmtHour(h.time)}</div>
      <div class="h-icon">${iconFor(h.condition)}</div>
      <div class="h-temp">${Math.round(h.temp_C)}°</div>
    `;
    hourlyStrip.appendChild(el);
  });

  // Daily list
  const dailyList = document.getElementById("dailyList");
  dailyList.innerHTML = "";
  data.daily_outlook.forEach((d, i) => {
    const row = document.createElement("div");
    row.className = "day-row";
    row.innerHTML = `
      <div class="d-day">${fmtDay(d.date, i)}</div>
      <div class="d-icon">${iconFor(d.condition)}</div>
      <div class="d-rain">${d.rain_mm > 0 ? d.rain_mm.toFixed(1) + " mm" : ""}</div>
      <div class="d-range">${Math.round(d.temp_min_C)}° - ${Math.round(d.temp_max_C)}°</div>
    `;
    dailyList.appendChild(row);
  });

  // Flood banner
  const floodBanner = document.getElementById("floodBanner");
  if (data.flood_risk_available && data.flood_risk) {
    floodBanner.hidden = false;
    const fr = data.flood_risk;
    const bandEl = document.getElementById("floodBand");
    bandEl.textContent = `${fr.risk_band} (${Math.round(fr.flood_risk_probability * 100)}%)`;
    bandEl.className = "flood-band " + fr.risk_band;
    document.getElementById("floodSub").textContent =
      `Most recent real data: ${fr.most_recent_real_data_date}`;
    const why = document.getElementById("floodWhy");
    why.innerHTML = fr.explanation
      .map(e => `<div><span class="${e.direction}">${e.direction === "up" ? "↑" : "↓"} ${e.feature}</span></div>`)
      .join("");
  } else {
    floodBanner.hidden = true;
  }

  // Map
  initMap(loc.lat, loc.lon, nameParts.join(", "));
}

// ---------- Fetching ----------

async function loadByCoords(lat, lon, name, admin1, country) {
  setStatus(`Loading weather for ${name || "your location"}…`);
  try {
    const params = new URLSearchParams({ lat, lon });
    if (name) params.set("name", name);
    if (admin1) params.set("admin1", admin1);
    if (country) params.set("country", country);
    const resp = await fetch(`/weather-by-coords?${params.toString()}`);
    if (!resp.ok) throw new Error((await resp.json()).detail || "Request failed");
    render(await resp.json());
  } catch (err) {
    setStatus(`Couldn't load weather: ${err.message}`, true);
  }
}

async function loadByName(query) {
  setStatus(`Looking up "${query}"…`);
  try {
    const resp = await fetch(`/weather/${encodeURIComponent(query)}`);
    if (!resp.ok) throw new Error((await resp.json()).detail || "Request failed");
    render(await resp.json());
  } catch (err) {
    setStatus(`Couldn't load weather: ${err.message}`, true);
  }
}

// ---------- Search autocomplete ----------

let debounceTimer = null;

function hideSuggestions() {
  suggestionsEl.hidden = true;
  suggestionsEl.innerHTML = "";
  activeSuggestionIndex = -1;
  currentSuggestions = [];
}

function renderSuggestions(results) {
  currentSuggestions = results;
  activeSuggestionIndex = -1;
  if (!results.length) {
    suggestionsEl.innerHTML = '<li class="no-results">No matches</li>';
    suggestionsEl.hidden = false;
    return;
  }
  suggestionsEl.innerHTML = results
    .map((r, i) => {
      const meta = [r.admin1, r.country].filter(Boolean).join(", ");
      return `<li data-index="${i}"><span class="s-name">${r.name}</span><span class="s-meta">${meta}</span></li>`;
    })
    .join("");
  suggestionsEl.hidden = false;
}

searchInput.addEventListener("input", () => {
  const q = searchInput.value.trim();
  clearTimeout(debounceTimer);
  if (q.length < 2) {
    hideSuggestions();
    return;
  }
  debounceTimer = setTimeout(async () => {
    try {
      const resp = await fetch(`/geocode-search?q=${encodeURIComponent(q)}`);
      const data = await resp.json();
      renderSuggestions(data.results || []);
    } catch (err) {
      hideSuggestions();
    }
  }, 300);
});

suggestionsEl.addEventListener("click", (e) => {
  const li = e.target.closest("li[data-index]");
  if (!li) return;
  const r = currentSuggestions[Number(li.dataset.index)];
  searchInput.value = r.name;
  hideSuggestions();
  loadByCoords(r.lat, r.lon, r.name, r.admin1, r.country);
});

searchInput.addEventListener("keydown", (e) => {
  const items = suggestionsEl.querySelectorAll("li[data-index]");
  if (!items.length) return;
  if (e.key === "ArrowDown") {
    e.preventDefault();
    activeSuggestionIndex = Math.min(activeSuggestionIndex + 1, items.length - 1);
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    activeSuggestionIndex = Math.max(activeSuggestionIndex - 1, 0);
  } else if (e.key === "Enter" && activeSuggestionIndex >= 0) {
    e.preventDefault();
    items[activeSuggestionIndex].dispatchEvent(new MouseEvent("click", { bubbles: true }));
    return;
  } else {
    return;
  }
  items.forEach((it, i) => it.classList.toggle("active", i === activeSuggestionIndex));
});

document.addEventListener("click", (e) => {
  if (!e.target.closest(".search-wrap")) hideSuggestions();
});

searchForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = searchInput.value.trim();
  if (query) {
    hideSuggestions();
    loadByName(query);
  }
});

function useMyLocation() {
  if (!navigator.geolocation) {
    setStatus("Your browser doesn't support location detection. Search for a city instead.", true);
    return;
  }
  setStatus("Getting your location…");
  navigator.geolocation.getCurrentPosition(
    (pos) => loadByCoords(pos.coords.latitude, pos.coords.longitude, "Current Location", "", ""),
    () => loadByName("Kodagu"),
    { timeout: 8000 }
  );
}

locateBtn.addEventListener("click", useMyLocation);

// First load: try real GPS; if denied/unavailable, fall back to a default
// search so the page is never blank.
useMyLocation();

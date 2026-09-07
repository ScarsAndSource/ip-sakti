// config.js — include this before any other script on every page.
window.API_BASE = window.API_BASE || "https://ip-sakti-vp8d.onrender.com";

window.IPSAKTI = {
  getJurisdiction() {
    return localStorage.getItem("ipsakti_jurisdiction") || "india";
  },
  setJurisdiction(j) {
    localStorage.setItem("ipsakti_jurisdiction", j);
  },
  getClassification() {
    const raw = sessionStorage.getItem("ipsakti_classification");
    return raw ? JSON.parse(raw) : null;
  },
  setClassification(obj) {
    sessionStorage.setItem("ipsakti_classification", JSON.stringify(obj));
  },
  clearClassification() {
    sessionStorage.removeItem("ipsakti_classification");
  },
  categoryLabel(cat) {
    return {
      classical_asu: "Classical ASU Drug",
      patent_proprietary_asu: "Patent / Proprietary ASU Medicine",
      phytopharmaceutical: "Phytopharmaceutical",
      cosmetic: "Cosmetic",
      nutraceutical: "Nutraceutical",
      unclassified: "Unclassified — Needs Review",
    }[cat] || cat;
  },
  confidenceBand(score) {
    // Mirrors backend CONFIDENCE_THRESHOLD default (0.55). If you change
    // CONFIDENCE_THRESHOLD in .env, update the 0.55 below to match.
    if (score >= 0.75) return "High";
    if (score >= 0.55) return "Medium";
    return "Low";
  },
};

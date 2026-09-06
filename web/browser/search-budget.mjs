// Versioned search limits. Wall time, including inference, remains authoritative.
export const SEARCH_BUDGET_VERSION = 2;
export function searchNodeCap(seconds) {
  return Math.min(30000000, Math.max(15000, Math.round(seconds * 1000000)));
}
export function mainNodeCap(seconds) {
  return Math.min(1500000, Math.max(5000, Math.floor(seconds * 50000)));
}

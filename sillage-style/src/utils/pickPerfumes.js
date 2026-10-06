/**
 * Подбирает позиции из каталога по пересечению с ranked семействами.
 * Жёстко выкидывает ароматы, у которых в families есть любое из excludedFamilies
 * (как «Без цитруса» на основной панели).
 */
export function pickPerfumes(ranked, perfumes, { limit = 8, excludedFamilies = [] } = {}) {
  if (!ranked?.length || !perfumes?.length) return [];

  const excluded = new Set(excludedFamilies);
  const famSet = new Set(ranked.map((r) => r.family));
  const scored = [];
  for (const p of perfumes) {
    const fams = Array.isArray(p.families) ? p.families : [];
    if (fams.some((f) => excluded.has(f))) continue;
    if (!fams.some((f) => famSet.has(f))) continue;
    let s = 0;
    for (const r of ranked) {
      if (fams.includes(r.family)) s += r.score;
    }
    scored.push({ ...p, matchScore: s });
  }
  scored.sort((a, b) => b.matchScore - a.matchScore || (b.families?.length || 0) - (a.families?.length || 0));
  return scored.slice(0, limit);
}
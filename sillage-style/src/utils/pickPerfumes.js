/**
 * Подбирает позиции из каталога по пересечению с ranked семействами.
 * Дополнительно сохраняет совпавшие семейства, чтобы объяснить выбор аромата.
 */
export function pickPerfumes(ranked, perfumes, { limit = 8, excludedFamilies = [] } = {}) {
  if (!ranked?.length || !perfumes?.length) return [];

  const excluded = new Set(excludedFamilies);
  const famSet = new Set(ranked.map((r) => r.family));
  const scored = [];

  for (const p of perfumes) {
    const fams = Array.isArray(p.families) ? p.families : [];
    if (fams.some((f) => excluded.has(f))) continue;

    const matchedFamilies = ranked
      .filter((r) => fams.includes(r.family) && famSet.has(r.family))
      .map((r) => ({ family: r.family, score: r.score }));

    if (matchedFamilies.length === 0) continue;

    const matchScore = matchedFamilies.reduce((sum, item) => sum + item.score, 0);

    scored.push({
      ...p,
      matchScore,
      matchedFamilies,
    });
  }

  scored.sort(
    (a, b) =>
      b.matchScore - a.matchScore ||
      (b.matchedFamilies?.length || 0) - (a.matchedFamilies?.length || 0)
  );

  return scored.slice(0, limit);
}

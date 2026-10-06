/**
 * Суммирует веса по выбранным тегам; победитель только среди семейств,
 * не попавших в excluded (hard constraints).
 */
export function computeScores(matrix, activeTagNames, excludedFamilies) {
  const { families, tags } = matrix;
  const excluded = new Set(excludedFamilies);
  const byTag = new Map(tags.map((t) => [t.tag, t.scores]));

  const totals = Object.fromEntries(families.map((f) => [f, 0]));
  for (const name of activeTagNames) {
    const row = byTag.get(name);
    if (!row) continue;
    for (const f of families) {
      totals[f] += row[f] ?? 0;
    }
  }

  const ranked = families
    .filter((f) => !excluded.has(f))
    .map((f) => ({ family: f, score: totals[f] }))
    .sort((a, b) => b.score - a.score);

  return { totals, ranked, winner: ranked[0] ?? null };
}

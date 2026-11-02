/**
 * Суммирует веса по выбранным факторам (цвет, стиль, сезон).
 * Одновременно сохраняет вклад каждого фактора, чтобы потом объяснить результат.
 */
export function computeScores(matrix, activeTagNames, excludedFamilies) {
  const { families, tags } = matrix;
  const excluded = new Set(excludedFamilies);
  const byTag = new Map(tags.map((t) => [t.tag, t]));

  const totals = Object.fromEntries(families.map((f) => [f, 0]));
  const breakdown = Object.fromEntries(families.map((f) => [f, []]));

  for (const name of activeTagNames) {
    const row = byTag.get(name);
    if (!row) continue;

    for (const family of families) {
      const value = row.scores?.[family] ?? 0;
      totals[family] += value;
      breakdown[family].push({
        tag: row.tag,
        categoryRu: row.categoryRu || "Фактор",
        score: value,
      });
    }
  }

  const ranked = families
    .filter((f) => !excluded.has(f))
    .map((f) => ({
      family: f,
      score: totals[f],
      factors: breakdown[f],
    }))
    .sort((a, b) => b.score - a.score);

  const winner = ranked[0] ?? null;

  const explanation = winner
    ? {
        family: winner.family,
        totalScore: winner.score,
        factors: [...winner.factors].sort((a, b) => b.score - a.score),
        positive: winner.factors
          .filter((item) => item.score > 0)
          .sort((a, b) => b.score - a.score),
        negative: winner.factors
          .filter((item) => item.score < 0)
          .sort((a, b) => a.score - b.score),
      }
    : null;

  return { totals, breakdown, ranked, winner, explanation };
}

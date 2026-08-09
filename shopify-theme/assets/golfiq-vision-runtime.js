(() => {
  'use strict';

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  function validateResult(result) {
    if (!result || typeof result !== 'object') return { ok: false, reason: 'invalid_result' };
    if (!['ready', 'direction_only', 'withheld'].includes(result.status)) {
      return { ok: false, reason: 'invalid_status' };
    }
    const confidence = Number(result.confidence);
    if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
      return { ok: false, reason: 'invalid_confidence' };
    }
    if (result.status === 'ready') {
      const low = Number(result.carry_low_yards);
      const mid = Number(result.carry_mid_yards);
      const high = Number(result.carry_high_yards);
      if (![low, mid, high].every(Number.isFinite) || !(low <= mid && mid <= high)) {
        return { ok: false, reason: 'invalid_carry_range' };
      }
    }
    return { ok: true };
  }

  function publicShot(result) {
    const validity = validateResult(result);
    if (!validity.ok) {
      return { status: 'withheld', confidence: 0, reasons: [validity.reason] };
    }
    const safe = {
      status: result.status,
      direction: result.direction || 'unavailable',
      shape: result.shape || 'unavailable',
      confidence: clamp(Number(result.confidence), 0, 1),
      reasons: Array.isArray(result.reasons) ? result.reasons : []
    };
    if (result.status === 'ready' && safe.confidence >= 0.65) {
      safe.carry_low_yards = Math.round(Number(result.carry_low_yards));
      safe.carry_mid_yards = Math.round(Number(result.carry_mid_yards));
      safe.carry_high_yards = Math.round(Number(result.carry_high_yards));
    } else if (result.status === 'ready') {
      safe.status = 'direction_only';
      safe.reasons = [...safe.reasons, 'carry_confidence_below_release_threshold'];
    }
    return safe;
  }

  function registerCourseLanding(result) {
    const shot = publicShot(result);
    if (shot.status !== 'ready' || !window.GolfIQCourse?.registerLanding) return false;
    window.GolfIQCourse.registerLanding({
      carryYards: shot.carry_mid_yards,
      lateralYards: Number(result.lateral_yards || 0),
      shape: shot.shape,
      confidence: shot.confidence >= 0.8 ? 'high' : 'medium',
      source: 'golfiq-vision'
    });
    return true;
  }

  window.GolfIQVision = Object.freeze({ validateResult, publicShot, registerCourseLanding });
})();

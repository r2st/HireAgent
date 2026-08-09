"""Availability computation for interview scheduling (design §4.3).

Finding a slot is interval arithmetic: take each interviewer's working hours,
subtract what their calendar says they are busy with, intersect what is left
across everyone who must attend, then chop the survivors into bookable slots.

Everything here is pure and works in UTC. Local wall-clock times only exist
while working hours are being expanded — a day at a time, through the
interviewer's own ``ZoneInfo`` — because "09:00–17:00 Mon–Fri" is a statement
about local time that shifts against UTC twice a year. Expanding per local day
rather than adding a fixed offset is what keeps slots at 9am local across a DST
boundary instead of drifting to 8am or 10am.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)

# Index matches ``datetime.weekday()``: Monday is 0.
DAY_KEYS: tuple[str, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

# Used when a calendar account has not configured its own hours.
DEFAULT_WORKING_HOURS: dict[str, list[list[str]]] = {
    "mon": [["09:00", "17:00"]],
    "tue": [["09:00", "17:00"]],
    "wed": [["09:00", "17:00"]],
    "thu": [["09:00", "17:00"]],
    "fri": [["09:00", "17:00"]],
}

# Slot starts are aligned to this grid so candidates are offered 10:00 and
# 10:30 rather than 10:07.
DEFAULT_GRANULARITY_MINUTES = 30

# A slot must be at least this far out, so nobody is offered an interview
# starting in ten minutes.
DEFAULT_MIN_NOTICE_HOURS = 12

# How far ahead to look when nobody says otherwise.
DEFAULT_HORIZON_DAYS = 14


@dataclass(frozen=True, order=True)
class Interval:
    """A half-open UTC time range ``[start, end)``.

    Half-open is what makes back-to-back intervals not overlap, so a meeting
    ending at 10:00 and one starting at 10:00 are a legal pair.
    """

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("Interval bounds must be timezone-aware")
        if self.end <= self.start:
            raise ValueError(f"Interval end {self.end} is not after start {self.start}")

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    @property
    def minutes(self) -> float:
        return self.duration.total_seconds() / 60

    def overlaps(self, other: Interval) -> bool:
        return self.start < other.end and other.start < self.end

    def contains(self, other: Interval) -> bool:
        return self.start <= other.start and other.end <= self.end

    def to_dict(self) -> dict[str, str]:
        return {"start": to_iso(self.start), "end": to_iso(self.end)}


def to_utc(value: datetime) -> datetime:
    """Coerce a datetime to UTC, treating naive input as already-UTC.

    Naive datetimes reach us from SQLite, which drops tzinfo on the way back
    out of a ``DateTime(timezone=True)`` column.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def to_iso(value: datetime) -> str:
    """ISO-8601 in UTC with a ``Z`` suffix, which is what the API emits."""
    return to_utc(value).isoformat().replace("+00:00", "Z")


def load_zone(name: str | None) -> ZoneInfo:
    """Resolve a timezone name, falling back to UTC rather than raising.

    A bad timezone string in a calendar account is a data problem; it should
    degrade the slots offered, not take out the scheduling endpoint.
    """
    if not name:
        return ZoneInfo("UTC")
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        logger.warning("Unknown timezone %r; falling back to UTC", name)
        return ZoneInfo("UTC")


# --------------------------------------------------------------------------- #
# Interval algebra
# --------------------------------------------------------------------------- #
def merge(intervals: list[Interval]) -> list[Interval]:
    """Sort and coalesce overlapping or touching intervals."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda i: (i.start, i.end))
    merged = [ordered[0]]
    for current in ordered[1:]:
        last = merged[-1]
        # ``<=`` so [9,10) and [10,11) coalesce into [9,11): for busy time,
        # two back-to-back meetings are one continuous block of unavailability.
        if current.start <= last.end:
            if current.end > last.end:
                merged[-1] = Interval(last.start, current.end)
        else:
            merged.append(current)
    return merged


def subtract(base: list[Interval], blocks: list[Interval]) -> list[Interval]:
    """Remove ``blocks`` from ``base``, returning what is left."""
    if not base:
        return []
    busy = merge(blocks)
    if not busy:
        return merge(base)

    remaining: list[Interval] = []
    for span in merge(base):
        cursor = span.start
        for block in busy:
            if block.end <= cursor:
                continue
            if block.start >= span.end:
                break
            if block.start > cursor:
                remaining.append(Interval(cursor, block.start))
            cursor = max(cursor, block.end)
            if cursor >= span.end:
                break
        if cursor < span.end:
            remaining.append(Interval(cursor, span.end))
    return remaining


def intersect(a: list[Interval], b: list[Interval]) -> list[Interval]:
    """Time present in both lists.

    A linear two-pointer sweep rather than a nested loop, because a panel
    interview intersects one list per interviewer and the lists are already
    sorted by ``merge``.
    """
    left, right = merge(a), merge(b)
    result: list[Interval] = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i].start, right[j].start)
        end = min(left[i].end, right[j].end)
        if start < end:
            result.append(Interval(start, end))
        # Advance whichever interval ends first; the other may still overlap
        # the next one along.
        if left[i].end < right[j].end:
            i += 1
        else:
            j += 1
    return result


def intersect_all(groups: list[list[Interval]]) -> list[Interval]:
    """Time common to every group. An empty group makes the result empty."""
    if not groups:
        return []
    common = merge(groups[0])
    for group in groups[1:]:
        common = intersect(common, group)
        if not common:
            break
    return common


# --------------------------------------------------------------------------- #
# Working hours -> UTC intervals
# --------------------------------------------------------------------------- #
def parse_hhmm(value: str) -> time:
    """Parse ``"09:00"`` / ``"9:00"`` / ``"09:00:00"`` into a ``time``."""
    parts = str(value).strip().split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid time-of-day {value!r}; expected HH:MM")
    hour, minute = int(parts[0]), int(parts[1])
    second = int(parts[2]) if len(parts) > 2 else 0
    if not (0 <= hour <= 24 and 0 <= minute < 60 and 0 <= second < 60):
        raise ValueError(f"Out-of-range time-of-day {value!r}")
    return time(hour=hour % 24, minute=minute, second=second)


def expand_working_hours(
    working_hours: dict | None,
    timezone_name: str,
    window: Interval,
) -> list[Interval]:
    """Project weekly working hours onto a concrete UTC window.

    ``working_hours`` maps a day key to a list of ``[start, end]`` local
    wall-clock pairs, e.g. ``{"mon": [["09:00", "13:00"], ["14:00", "18:00"]]}``
    — the split pair being how a lunch break is expressed.
    """
    schedule = working_hours if working_hours else DEFAULT_WORKING_HOURS
    zone = load_zone(timezone_name)

    # Walk local dates, not UTC dates: the UTC window can straddle three local
    # days once an offset like +13:00 is involved, and a day missed here is a
    # day of availability silently lost.
    first_local: date = window.start.astimezone(zone).date() - timedelta(days=1)
    last_local: date = window.end.astimezone(zone).date() + timedelta(days=1)

    out: list[Interval] = []
    current = first_local
    while current <= last_local:
        for start_at, end_at in _ranges_for_day(schedule, current):
            local_start = datetime.combine(current, start_at, tzinfo=zone)
            # "24:00" and "00:00" both mean the end of the day when they close
            # a range, so roll onto the next date rather than producing an
            # empty or inverted interval.
            end_date = current + timedelta(days=1) if end_at == time(0, 0) else current
            local_end = datetime.combine(end_date, end_at, tzinfo=zone)
            if local_end <= local_start:
                continue
            clipped = _clip(Interval(to_utc(local_start), to_utc(local_end)), window)
            if clipped is not None:
                out.append(clipped)
        current += timedelta(days=1)
    return merge(out)


def _ranges_for_day(schedule: dict, day: date) -> list[tuple[time, time]]:
    """The local ranges configured for one weekday, skipping malformed entries."""
    raw = schedule.get(DAY_KEYS[day.weekday()]) or []
    if not isinstance(raw, list):
        return []
    ranges: list[tuple[time, time]] = []
    for entry in raw:
        if not isinstance(entry, list | tuple) or len(entry) != 2:
            logger.warning("Skipping malformed working-hours entry %r", entry)
            continue
        try:
            ranges.append((parse_hhmm(entry[0]), parse_hhmm(entry[1])))
        except (ValueError, TypeError):
            logger.warning("Skipping unparseable working-hours entry %r", entry)
    return ranges


def _clip(interval: Interval, window: Interval) -> Interval | None:
    start = max(interval.start, window.start)
    end = min(interval.end, window.end)
    return Interval(start, end) if start < end else None


# --------------------------------------------------------------------------- #
# Slot generation
# --------------------------------------------------------------------------- #
def align_up(value: datetime, granularity_minutes: int) -> datetime:
    """Round ``value`` up onto the granularity grid, measured from the hour."""
    if granularity_minutes <= 0:
        return value
    step = timedelta(minutes=granularity_minutes)
    hour_start = value.replace(minute=0, second=0, microsecond=0)
    elapsed = value - hour_start
    steps = -(-elapsed // step)  # ceiling division on timedeltas
    return hour_start + steps * step


def slice_slots(
    intervals: list[Interval],
    *,
    duration_minutes: int,
    granularity_minutes: int = DEFAULT_GRANULARITY_MINUTES,
    buffer_minutes: int = 0,
) -> list[Interval]:
    """Chop free intervals into bookable slots of a fixed length.

    ``buffer_minutes`` is padding that must also be free *after* the meeting,
    so an interviewer is not booked to the second between two calls. The buffer
    is not part of the returned slot — the candidate is offered the interview,
    not the breathing room.
    """
    if duration_minutes <= 0:
        raise ValueError("duration_minutes must be positive")

    span = timedelta(minutes=duration_minutes + max(0, buffer_minutes))
    length = timedelta(minutes=duration_minutes)

    slots: list[Interval] = []
    for interval in merge(intervals):
        cursor = align_up(interval.start, granularity_minutes)
        while cursor + span <= interval.end:
            slots.append(Interval(cursor, cursor + length))
            cursor += timedelta(minutes=granularity_minutes or duration_minutes)
    return slots


def spread_slots(slots: list[Interval], limit: int, *, per_day: int = 3) -> list[Interval]:
    """Trim to ``limit`` slots while keeping them spread across days.

    Offering a candidate the first N openings hands them five consecutive
    Monday-morning slots and nothing else; capping per day gives them a real
    choice of day, which is the point of proposing slots at all.
    """
    if limit <= 0 or not slots:
        return []

    ordered = sorted(slots, key=lambda s: s.start)
    if limit >= len(ordered):
        return ordered

    # Widen the per-day cap until the limit can be met, so a single available
    # day still fills the offer rather than returning ``per_day`` slots. The
    # busiest day bounds the search: at that cap nothing is being filtered out
    # any more, so widening further cannot add a slot.
    busiest = max(Counter(s.start.date() for s in ordered).values())
    picked: list[Interval] = []
    for cap in range(max(1, per_day), busiest + 1):
        seen: dict[date, int] = {}
        picked = []
        for slot in ordered:
            day = slot.start.date()
            if seen.get(day, 0) >= cap:
                continue
            seen[day] = seen.get(day, 0) + 1
            picked.append(slot)
            if len(picked) == limit:
                return picked
    return picked[:limit]


# --------------------------------------------------------------------------- #
# High-level entry point
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParticipantAvailability:
    """One interviewer's free time, plus whether we trust it.

    ``calendar_synced`` is false when the calendar could not be read, meaning
    the free time is working hours alone. Callers surface that so a recruiter
    knows the slots may collide with something the interviewer has booked.
    """

    user_id: object
    free: list[Interval]
    calendar_synced: bool = True
    error: str | None = None


def find_slots(
    participants: list[ParticipantAvailability],
    window: Interval,
    *,
    duration_minutes: int,
    granularity_minutes: int = DEFAULT_GRANULARITY_MINUTES,
    buffer_minutes: int = 0,
    min_notice_hours: int = DEFAULT_MIN_NOTICE_HOURS,
    now: datetime | None = None,
    limit: int = 10,
    per_day: int = 3,
) -> list[Interval]:
    """Slots every participant can make, within ``window``.

    With no participants there is nobody to be busy, so the whole window is
    fair game — that is the right answer for a candidate-only async screening.
    """
    now = to_utc(now or datetime.now(UTC))
    earliest = max(window.start, now + timedelta(hours=max(0, min_notice_hours)))
    if earliest >= window.end:
        return []
    bounded = Interval(earliest, window.end)

    if participants:
        common = intersect_all([p.free for p in participants])
    else:
        common = [bounded]

    common = [c for c in (_clip(i, bounded) for i in common) if c is not None]
    slots = slice_slots(
        common,
        duration_minutes=duration_minutes,
        granularity_minutes=granularity_minutes,
        buffer_minutes=buffer_minutes,
    )
    return spread_slots(slots, limit, per_day=per_day)

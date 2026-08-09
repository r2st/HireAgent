"""The interval algebra behind slot proposal (design §4.3).

These are pure-function tests with no database: if the arithmetic here is
wrong, every scheduling feature above it is wrong in a way that only shows up
as a double-booked interviewer.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.services.availability import (
    DEFAULT_WORKING_HOURS,
    Interval,
    ParticipantAvailability,
    align_up,
    expand_working_hours,
    find_slots,
    intersect,
    intersect_all,
    load_zone,
    merge,
    parse_hhmm,
    slice_slots,
    spread_slots,
    subtract,
    to_iso,
    to_utc,
)


def dt(day: int, hour: int, minute: int = 0) -> datetime:
    """A UTC datetime in January 2027 — a Friday-anchored, DST-free month."""
    return datetime(2027, 1, day, hour, minute, tzinfo=UTC)


def iv(day: int, start_hour: float, end_hour: float) -> Interval:
    def to_dt(value: float) -> datetime:
        hour = int(value)
        return dt(day, hour, int(round((value - hour) * 60)))

    return Interval(to_dt(start_hour), to_dt(end_hour))


class TestInterval:
    def test_rejects_a_naive_datetime(self) -> None:
        # A naive bound would silently compare wrong against UTC-aware ones.
        with pytest.raises(ValueError, match="timezone-aware"):
            Interval(datetime(2027, 1, 4, 9), dt(4, 10))

    def test_rejects_an_inverted_range(self) -> None:
        with pytest.raises(ValueError, match="not after start"):
            Interval(dt(4, 10), dt(4, 9))

    def test_rejects_a_zero_length_range(self) -> None:
        with pytest.raises(ValueError):
            Interval(dt(4, 10), dt(4, 10))

    def test_back_to_back_intervals_do_not_overlap(self) -> None:
        # Half-open ranges: a meeting ending at 10:00 and one starting at
        # 10:00 are a legal pair, not a conflict.
        assert not iv(4, 9, 10).overlaps(iv(4, 10, 11))

    def test_overlap_is_symmetric(self) -> None:
        a, b = iv(4, 9, 11), iv(4, 10, 12)
        assert a.overlaps(b) and b.overlaps(a)

    def test_contains(self) -> None:
        assert iv(4, 9, 12).contains(iv(4, 10, 11))
        assert not iv(4, 10, 11).contains(iv(4, 9, 12))

    def test_minutes(self) -> None:
        assert iv(4, 9, 10.5).minutes == 90

    def test_to_dict_emits_z_suffixed_iso(self) -> None:
        assert iv(4, 9, 10).to_dict() == {
            "start": "2027-01-04T09:00:00Z",
            "end": "2027-01-04T10:00:00Z",
        }


class TestCoercion:
    def test_naive_datetimes_are_treated_as_utc(self) -> None:
        # SQLite hands back naive datetimes from timezone-aware columns.
        assert to_utc(datetime(2027, 1, 4, 9)) == dt(4, 9)

    def test_aware_datetimes_are_converted_not_relabelled(self) -> None:
        kolkata = datetime(2027, 1, 4, 14, 30, tzinfo=load_zone("Asia/Kolkata"))
        assert to_utc(kolkata) == dt(4, 9)

    def test_unknown_timezone_falls_back_to_utc(self) -> None:
        # A bad timezone string is a data problem, not a reason to 500.
        assert load_zone("Mars/Olympus_Mons").key == "UTC"
        assert load_zone(None).key == "UTC"
        assert load_zone("").key == "UTC"

    def test_to_iso_normalises_to_utc(self) -> None:
        kolkata = datetime(2027, 1, 4, 14, 30, tzinfo=load_zone("Asia/Kolkata"))
        assert to_iso(kolkata) == "2027-01-04T09:00:00Z"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("09:00", (9, 0)), ("9:00", (9, 0)), ("09:30:00", (9, 30)), ("24:00", (0, 0))],
    )
    def test_parse_hhmm(self, raw: str, expected: tuple[int, int]) -> None:
        parsed = parse_hhmm(raw)
        assert (parsed.hour, parsed.minute) == expected

    @pytest.mark.parametrize("raw", ["nonsense", "9", "25:00", "09:75"])
    def test_parse_hhmm_rejects_junk(self, raw: str) -> None:
        with pytest.raises(ValueError):
            parse_hhmm(raw)


class TestMerge:
    def test_empty(self) -> None:
        assert merge([]) == []

    def test_overlapping_intervals_coalesce(self) -> None:
        assert merge([iv(4, 9, 11), iv(4, 10, 12)]) == [iv(4, 9, 12)]

    def test_touching_intervals_coalesce(self) -> None:
        # Two back-to-back meetings are one continuous block of busy time.
        assert merge([iv(4, 9, 10), iv(4, 10, 11)]) == [iv(4, 9, 11)]

    def test_disjoint_intervals_are_kept_and_sorted(self) -> None:
        assert merge([iv(4, 14, 15), iv(4, 9, 10)]) == [iv(4, 9, 10), iv(4, 14, 15)]

    def test_a_contained_interval_is_absorbed(self) -> None:
        assert merge([iv(4, 9, 17), iv(4, 10, 11)]) == [iv(4, 9, 17)]


class TestSubtract:
    def test_removing_the_middle_splits_the_interval(self) -> None:
        assert subtract([iv(4, 9, 17)], [iv(4, 12, 13)]) == [
            iv(4, 9, 12),
            iv(4, 13, 17),
        ]

    def test_removing_nothing_returns_the_base(self) -> None:
        assert subtract([iv(4, 9, 17)], []) == [iv(4, 9, 17)]

    def test_a_fully_covering_block_leaves_nothing(self) -> None:
        assert subtract([iv(4, 9, 17)], [iv(4, 8, 18)]) == []

    def test_a_non_overlapping_block_is_ignored(self) -> None:
        assert subtract([iv(4, 9, 17)], [iv(5, 9, 17)]) == [iv(4, 9, 17)]

    def test_leading_and_trailing_blocks_trim_the_edges(self) -> None:
        assert subtract([iv(4, 9, 17)], [iv(4, 8, 10), iv(4, 16, 18)]) == [
            iv(4, 10, 16)
        ]

    def test_overlapping_blocks_are_merged_before_subtraction(self) -> None:
        # Two overlapping meetings must not each punch their own hole.
        assert subtract([iv(4, 9, 17)], [iv(4, 11, 13), iv(4, 12, 14)]) == [
            iv(4, 9, 11),
            iv(4, 14, 17),
        ]

    def test_subtracting_from_nothing_yields_nothing(self) -> None:
        assert subtract([], [iv(4, 9, 10)]) == []


class TestIntersect:
    def test_partial_overlap(self) -> None:
        assert intersect([iv(4, 9, 12)], [iv(4, 11, 14)]) == [iv(4, 11, 12)]

    def test_no_overlap(self) -> None:
        assert intersect([iv(4, 9, 10)], [iv(4, 11, 12)]) == []

    def test_multiple_overlaps_are_all_found(self) -> None:
        # The two-pointer sweep must not stop at the first match.
        result = intersect(
            [iv(4, 9, 12), iv(4, 14, 17)], [iv(4, 11, 15), iv(4, 16, 18)]
        )
        assert result == [iv(4, 11, 12), iv(4, 14, 15), iv(4, 16, 17)]

    def test_intersect_all_with_one_group(self) -> None:
        assert intersect_all([[iv(4, 9, 12)]]) == [iv(4, 9, 12)]

    def test_intersect_all_narrows_across_groups(self) -> None:
        assert intersect_all(
            [[iv(4, 9, 17)], [iv(4, 10, 16)], [iv(4, 11, 12)]]
        ) == [iv(4, 11, 12)]

    def test_one_empty_group_empties_the_result(self) -> None:
        # An interviewer with no free time means the panel has no free time.
        assert intersect_all([[iv(4, 9, 17)], []]) == []

    def test_no_groups_is_empty(self) -> None:
        assert intersect_all([]) == []


class TestWorkingHours:
    def test_default_hours_cover_weekdays_only(self) -> None:
        # 2027-01-04 is a Monday; the window runs Monday to Sunday.
        window = Interval(dt(4, 0), dt(11, 0))
        intervals = expand_working_hours(None, "UTC", window)
        assert len(intervals) == 5
        assert all(i.minutes == 480 for i in intervals)
        assert intervals[0] == iv(4, 9, 17)

    def test_hours_are_local_and_converted_to_utc(self) -> None:
        window = Interval(dt(4, 0), dt(5, 0))
        intervals = expand_working_hours(
            {"mon": [["09:00", "17:00"]]}, "Asia/Kolkata", window
        )
        # 09:00 IST is 03:30 UTC.
        assert intervals == [Interval(dt(4, 3, 30), dt(4, 11, 30))]

    def test_a_split_day_models_a_lunch_break(self) -> None:
        window = Interval(dt(4, 0), dt(5, 0))
        intervals = expand_working_hours(
            {"mon": [["09:00", "13:00"], ["14:00", "18:00"]]}, "UTC", window
        )
        assert intervals == [iv(4, 9, 13), iv(4, 14, 18)]

    def test_hours_are_clipped_to_the_window(self) -> None:
        window = Interval(dt(4, 10), dt(4, 12))
        intervals = expand_working_hours({"mon": [["09:00", "17:00"]]}, "UTC", window)
        assert intervals == [iv(4, 10, 12)]

    def test_midnight_end_rolls_to_the_next_day(self) -> None:
        window = Interval(dt(4, 0), dt(6, 0))
        intervals = expand_working_hours({"mon": [["22:00", "00:00"]]}, "UTC", window)
        assert intervals == [Interval(dt(4, 22), dt(5, 0))]

    def test_malformed_entries_are_skipped_not_fatal(self) -> None:
        window = Interval(dt(4, 0), dt(5, 0))
        intervals = expand_working_hours(
            {"mon": [["nonsense", "17:00"], ["10:00"], "junk", ["09:00", "10:00"]]},
            "UTC",
            window,
        )
        assert intervals == [iv(4, 9, 10)]

    def test_an_inverted_range_is_dropped(self) -> None:
        window = Interval(dt(4, 0), dt(5, 0))
        assert expand_working_hours({"mon": [["17:00", "09:00"]]}, "UTC", window) == []

    def test_unknown_day_keys_are_ignored(self) -> None:
        window = Interval(dt(4, 0), dt(5, 0))
        assert expand_working_hours({"funday": [["09:00", "17:00"]]}, "UTC", window) == []

    def test_hours_stay_at_nine_local_across_a_dst_change(self) -> None:
        """The point of expanding per local day rather than by fixed offset.

        US DST began 2027-03-14. A 9am New York start is 14:00 UTC before it
        and 13:00 UTC after; a fixed-offset implementation drifts by an hour.
        """
        window = Interval(
            datetime(2027, 3, 11, 0, tzinfo=UTC), datetime(2027, 3, 17, 0, tzinfo=UTC)
        )
        intervals = expand_working_hours(
            DEFAULT_WORKING_HOURS, "America/New_York", window
        )
        starts = {i.start.date(): i.start.hour for i in intervals}
        assert starts[datetime(2027, 3, 11).date()] == 14  # EST
        assert starts[datetime(2027, 3, 15).date()] == 13  # EDT

    def test_a_far_eastern_timezone_still_gets_every_local_day(self) -> None:
        # +13:00 pushes local days well off the UTC ones; walking UTC dates
        # would drop one end of the range.
        window = Interval(dt(4, 0), dt(9, 0))
        intervals = expand_working_hours(
            {k: [["09:00", "17:00"]] for k in ("mon", "tue", "wed", "thu", "fri")},
            "Pacific/Auckland",
            window,
        )
        assert len(intervals) == 5


class TestAlignAndSlice:
    @pytest.mark.parametrize(
        ("minute", "expected"),
        [(0, 0), (1, 30), (29, 30), (30, 30), (31, 0), (59, 0)],
    )
    def test_align_up_to_the_half_hour(self, minute: int, expected: int) -> None:
        aligned = align_up(dt(4, 9, minute), 30)
        assert aligned.minute == expected
        assert aligned.hour == (10 if minute > 30 else 9)

    def test_align_up_is_a_no_op_on_the_grid(self) -> None:
        assert align_up(dt(4, 9, 0), 30) == dt(4, 9, 0)

    def test_slots_tile_the_interval(self) -> None:
        slots = slice_slots([iv(4, 9, 11)], duration_minutes=30)
        assert [s.start.hour * 60 + s.start.minute for s in slots] == [
            540,
            570,
            600,
            630,
        ]

    def test_a_slot_never_runs_past_the_free_interval(self) -> None:
        slots = slice_slots([iv(4, 9, 10)], duration_minutes=45)
        # Only 09:00-09:45 fits; 09:30-10:15 would overrun.
        assert len(slots) == 1
        assert slots[0] == Interval(dt(4, 9), dt(4, 9, 45))

    def test_a_buffer_reserves_time_after_the_meeting(self) -> None:
        # 30-minute meeting + 15-minute buffer needs 45 minutes of clear time,
        # but the returned slot is still only the 30 minutes.
        slots = slice_slots(
            [iv(4, 9, 10)], duration_minutes=30, granularity_minutes=30, buffer_minutes=15
        )
        assert len(slots) == 1
        assert slots[0].minutes == 30

    def test_an_interval_shorter_than_the_duration_yields_nothing(self) -> None:
        assert slice_slots([iv(4, 9, 9.5)], duration_minutes=45) == []

    def test_unaligned_intervals_snap_to_the_grid(self) -> None:
        slots = slice_slots([iv(4, 9.25, 11)], duration_minutes=30)
        assert slots[0].start == dt(4, 9, 30)

    def test_zero_duration_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            slice_slots([iv(4, 9, 17)], duration_minutes=0)


class TestSpreadSlots:
    def test_slots_are_spread_across_days_not_taken_front_to_back(self) -> None:
        # Five slots on Monday and five on Tuesday: naive truncation would
        # offer the candidate a Monday morning and nothing else.
        slots = [iv(4, h, h + 1) for h in range(9, 14)] + [
            iv(5, h, h + 1) for h in range(9, 14)
        ]
        picked = spread_slots(slots, limit=4, per_day=2)
        assert len({s.start.date() for s in picked}) == 2
        assert len(picked) == 4

    def test_a_single_available_day_still_fills_the_limit(self) -> None:
        # The per-day cap widens rather than short-changing the candidate.
        slots = [iv(4, h, h + 1) for h in range(9, 15)]
        assert len(spread_slots(slots, limit=4, per_day=2)) == 4

    def test_fewer_slots_than_the_limit_returns_all_of_them(self) -> None:
        slots = [iv(4, 9, 10), iv(5, 9, 10)]
        assert spread_slots(slots, limit=10, per_day=3) == slots

    def test_zero_limit_returns_nothing(self) -> None:
        assert spread_slots([iv(4, 9, 10)], limit=0) == []

    def test_empty_input(self) -> None:
        assert spread_slots([], limit=5) == []


class TestFindSlots:
    def _participant(self, *intervals: Interval) -> ParticipantAvailability:
        return ParticipantAvailability(user_id="u", free=list(intervals))

    def test_slots_need_every_participant_free(self) -> None:
        window = Interval(dt(4, 0), dt(5, 0))
        slots = find_slots(
            [self._participant(iv(4, 9, 12)), self._participant(iv(4, 11, 17))],
            window,
            duration_minutes=60,
            min_notice_hours=0,
            now=dt(1, 0),
        )
        assert slots == [iv(4, 11, 12)]

    def test_min_notice_excludes_imminent_slots(self) -> None:
        window = Interval(dt(4, 0), dt(5, 0))
        slots = find_slots(
            [self._participant(iv(4, 9, 17))],
            window,
            duration_minutes=60,
            min_notice_hours=12,
            now=dt(4, 6),
        )
        # now + 12h = 18:00, which is past the free window entirely.
        assert slots == []

    def test_min_notice_trims_the_front_of_the_window(self) -> None:
        window = Interval(dt(4, 0), dt(5, 0))
        slots = find_slots(
            [self._participant(iv(4, 9, 17))],
            window,
            duration_minutes=60,
            min_notice_hours=4,
            now=dt(4, 8),
        )
        assert slots[0].start == dt(4, 12)

    def test_no_participants_means_the_whole_window_is_open(self) -> None:
        # An async video screening has no interviewer to be busy. Starts step
        # by the 30-minute granularity, so a 3-hour window yields five 1-hour
        # options rather than three back-to-back ones.
        window = Interval(dt(4, 9), dt(4, 12))
        slots = find_slots(
            [], window, duration_minutes=60, min_notice_hours=0, now=dt(1, 0)
        )
        assert [s.start for s in slots] == [
            dt(4, 9),
            dt(4, 9, 30),
            dt(4, 10),
            dt(4, 10, 30),
            dt(4, 11),
        ]

    def test_a_window_entirely_inside_the_notice_period_yields_nothing(self) -> None:
        window = Interval(dt(4, 9), dt(4, 17))
        assert (
            find_slots(
                [self._participant(iv(4, 9, 17))],
                window,
                duration_minutes=30,
                min_notice_hours=48,
                now=dt(4, 0),
            )
            == []
        )

    def test_the_limit_is_respected(self) -> None:
        window = Interval(dt(4, 0), dt(11, 0))
        slots = find_slots(
            [
                ParticipantAvailability(
                    user_id="u",
                    free=[iv(day, 9, 17) for day in range(4, 9)],
                )
            ],
            window,
            duration_minutes=30,
            min_notice_hours=0,
            now=dt(1, 0),
            limit=6,
            per_day=2,
        )
        assert len(slots) == 6
        assert len({s.start.date() for s in slots}) == 3

    def test_slots_come_back_in_chronological_order(self) -> None:
        window = Interval(dt(4, 0), dt(11, 0))
        slots = find_slots(
            [ParticipantAvailability(user_id="u", free=[iv(d, 9, 12) for d in (6, 4, 5)])],
            window,
            duration_minutes=60,
            min_notice_hours=0,
            now=dt(1, 0),
            limit=9,
        )
        assert slots == sorted(slots, key=lambda s: s.start)


class TestRegressionShapes:
    """Cases drawn from how real calendars misbehave."""

    def test_a_meeting_spanning_midnight_is_subtracted_correctly(self) -> None:
        free = [iv(4, 9, 17), iv(5, 9, 17)]
        overnight = [Interval(dt(4, 16), dt(5, 10))]
        assert subtract(free, overnight) == [iv(4, 9, 16), iv(5, 10, 17)]

    def test_an_all_day_block_clears_the_day(self) -> None:
        free = [iv(4, 9, 17), iv(5, 9, 17)]
        all_day = [Interval(dt(4, 0), dt(5, 0))]
        assert subtract(free, all_day) == [iv(5, 9, 17)]

    def test_many_small_busy_blocks_leave_the_gaps(self) -> None:
        busy = [iv(4, h, h + 0.5) for h in range(9, 17)]
        free = subtract([iv(4, 9, 17)], busy)
        assert len(free) == 8
        assert all(i.minutes == 30 for i in free)

    def test_a_fully_booked_day_offers_no_slots(self) -> None:
        window = Interval(dt(4, 0), dt(5, 0))
        free = subtract([iv(4, 9, 17)], [iv(4, 9, 17)])
        slots = find_slots(
            [ParticipantAvailability(user_id="u", free=free)],
            window,
            duration_minutes=30,
            min_notice_hours=0,
            now=dt(1, 0),
        )
        assert slots == []

    def test_five_interviewers_narrow_to_the_one_common_hour(self) -> None:
        window = Interval(dt(4, 0), dt(5, 0))
        participants = [
            ParticipantAvailability(user_id=f"u{n}", free=[iv(4, 9 + n, 15 + n)])
            for n in range(5)
        ]
        slots = find_slots(
            participants,
            window,
            duration_minutes=60,
            min_notice_hours=0,
            now=dt(1, 0),
        )
        # Latest start is 13:00, earliest end is 15:00 -> a two-hour overlap,
        # which offers three 1-hour starts on the half-hour grid.
        assert [s.start for s in slots] == [dt(4, 13), dt(4, 13, 30), dt(4, 14)]

    def test_a_window_of_a_single_minute_produces_nothing(self) -> None:
        window = Interval(dt(4, 9), dt(4, 9) + timedelta(minutes=1))
        assert (
            find_slots(
                [ParticipantAvailability(user_id="u", free=[iv(4, 9, 17)])],
                window,
                duration_minutes=30,
                min_notice_hours=0,
                now=dt(1, 0),
            )
            == []
        )

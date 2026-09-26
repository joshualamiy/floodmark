<script lang="ts">
	import { api } from "$lib/api";
	import { PredictionStatus } from "$lib/types/image-processing";
	import { createInfiniteQuery, createQuery } from "@tanstack/svelte-query";
	import { parseDate, type CalendarDate } from "@internationalized/date";
	import CalendarDays from "@lucide/svelte/icons/calendar-days";
	import X from "@lucide/svelte/icons/x";
	import { Button } from "$components/ui/button";
	import { Spinner } from "$components/ui/spinner";
	import * as Popover from "$components/ui/popover";
	import { RangeCalendar } from "$components/ui/range-calendar";

	let { cameraId, cameraName }: { cameraId: string; cameraName: string } = $props();

	let selectedHistoryId = $state<string | null>(null);
	let dateRange = $state<
		{ start: CalendarDate | undefined; end: CalendarDate | undefined } | undefined
	>(undefined);
	let fromTime = $state("00:00");
	let toTime = $state("23:59");

	const minimumHistoryDate = parseDate("2026-09-25");
	const maximumHistoryDate = parseDate(new Date().toISOString().slice(0, 10));

	const historyBounds = $derived.by(() => {
		if (!dateRange?.start || !dateRange.end) return null;
		return {
			from: toBoundary(dateRange.start, fromTime),
			to: toBoundary(dateRange.end, toTime),
		};
	});
	const hasCompleteHistoryRange = $derived(Boolean(historyBounds));
	const hasIncompleteHistoryRange = $derived(Boolean(dateRange) && !hasCompleteHistoryRange);
	const hasValidHistoryRange = $derived(
		!hasCompleteHistoryRange ||
			(historyBounds !== null && new Date(historyBounds.from) <= new Date(historyBounds.to)),
	);

	const historyQuery = createInfiniteQuery(() => ({
		queryKey: [
			"cameras",
			cameraId,
			"history",
			dateRange?.start?.toString(),
			dateRange?.end?.toString(),
			fromTime,
			toTime,
		],
		queryFn: ({ pageParam }) =>
			api()
				.camera(cameraId)
				.history({
					cursor: pageParam,
					from: hasValidHistoryRange ? historyBounds?.from : undefined,
					to: hasValidHistoryRange ? historyBounds?.to : undefined,
				}),
		initialPageParam: null as string | null,
		getNextPageParam: (lastPage) => lastPage.nextCursor,
		enabled: Boolean(cameraId && !hasIncompleteHistoryRange && hasValidHistoryRange),
	}));

	const historyImage = createQuery(() => ({
		queryKey: ["cameras", cameraId, "history-image", selectedHistoryId],
		queryFn: () => api().camera(cameraId).snapshot(false, selectedHistoryId!),
		enabled: Boolean(selectedHistoryId),
	}));

	const historyEntries = $derived(historyQuery.data?.pages.flatMap((page) => page.entries) ?? []);
	const selectedHistoryEntry = $derived(
		historyEntries.find((entry) => entry.id === selectedHistoryId),
	);
	const historyUrl = $derived(historyImage.data?.url);
	const hasHistoryFeed = $derived(Boolean(historyUrl && !historyUrl.includes("/skipped")));

	function toBoundary(date: CalendarDate, time: string): string {
		return new Date(`${date.toString()}T${time}:00`).toISOString();
	}

	function formatHistoryDate(value: Date): string {
		return new Intl.DateTimeFormat("en-US", {
			month: "short",
			day: "numeric",
			hour: "numeric",
			minute: "2-digit",
		}).format(new Date(value));
	}

	function historyStatus(entry: (typeof historyEntries)[number]): string {
		if (entry.alertStatus === PredictionStatus.Flooded) return "Flooded";
		if (entry.alertStatus === PredictionStatus.Wet) return "Possible flooding";
		if (entry.alertStatus === PredictionStatus.Dry) return "Clear";
		return "Unprocessed";
	}

	function loadMoreHistory(event: Event) {
		const element = event.currentTarget as HTMLDivElement;
		if (
			element.scrollTop + element.clientHeight >= element.scrollHeight - 80 &&
			historyQuery.hasNextPage &&
			!historyQuery.isFetchingNextPage
		) {
			void historyQuery.fetchNextPage();
		}
	}
</script>

<div class="mt-6 border-t pt-5">
	<div class="flex items-center justify-between gap-3">
		<div>
			<h2 class="text-sm font-semibold">History</h2>
			<p class="mt-1 text-xs text-muted-foreground">Select a capture to view it.</p>
		</div>
		<Popover.Root>
			<Popover.Trigger>
				{#snippet child({ props })}
					<Button aria-label="Filter history by date" size="icon-sm" variant="outline" {...props}>
						<CalendarDays />
					</Button>
				{/snippet}
			</Popover.Trigger>
			<Popover.Content align="end" class="w-auto p-0">
				<RangeCalendar
					bind:value={dateRange}
					minValue={minimumHistoryDate}
					maxValue={maximumHistoryDate}
					disableDaysOutsideMonth
				/>
				<div class="grid grid-cols-2 gap-3 border-t p-3">
					<label class="flex flex-col gap-1 text-xs font-medium">
						From
						<input
							bind:value={fromTime}
							class="h-9 rounded-md border border-input bg-background px-2 text-sm font-normal outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
							aria-label="History start time"
							type="time"
						/>
					</label>
					<label class="flex flex-col gap-1 text-xs font-medium">
						To
						<input
							bind:value={toTime}
							class="h-9 rounded-md border border-input bg-background px-2 text-sm font-normal outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
							aria-label="History end time"
							type="time"
						/>
					</label>
				</div>
				{#if hasCompleteHistoryRange && !hasValidHistoryRange}
					<p class="px-3 pb-3 text-xs text-destructive">The start must be before the end.</p>
				{/if}
				{#if dateRange}
					<div class="flex justify-end border-t p-3">
						<Button size="sm" variant="ghost" onclick={() => (dateRange = undefined)}>
							<X data-icon="inline-start" />
							Clear dates
						</Button>
					</div>
				{/if}
			</Popover.Content>
		</Popover.Root>
	</div>

	{#if dateRange?.start && dateRange.end}
		<p class="mt-2 text-xs text-muted-foreground">
			{dateRange.start.toString()}
			{fromTime} to {dateRange.end.toString()}
			{toTime}
		</p>
	{/if}

	<div class="mt-3 max-h-64 overflow-y-auto pr-1" onscroll={loadMoreHistory}>
		{#if historyQuery.isPending}
			<div class="flex justify-center py-6"><Spinner /></div>
		{:else if historyQuery.isError}
			<p class="py-6 text-center text-xs text-destructive">Unable to load history.</p>
		{:else if historyEntries.length === 0}
			<p class="py-6 text-center text-xs text-muted-foreground">No history found.</p>
		{:else}
			<div class="flex flex-col gap-1">
				{#each historyEntries as entry (entry.id)}
					<Popover.Root>
						<div
							class="flex w-full items-center justify-between gap-3 rounded-xl px-3 py-2 text-left"
						>
							<span class="min-w-0">
								<span class="block text-xs font-medium">{formatHistoryDate(entry.capturedAt)}</span>
								<span class="mt-0.5 block text-xs text-muted-foreground"
									>{historyStatus(entry)}</span
								>
							</span>
							<Popover.Trigger
								class="rounded-4xl focus-visible:ring-3 focus-visible:ring-ring/50"
								onclick={() => (selectedHistoryId = entry.id)}
							>
								{#snippet child({ props })}
									<Button
										variant={entry.alertStatus === PredictionStatus.Flooded
											? "destructive"
											: entry.alertStatus === PredictionStatus.Wet
												? "warning"
												: "secondary"}
										{...props}
									>
										View
									</Button>
								{/snippet}
							</Popover.Trigger>
						</div>
						<Popover.Content align="end" class="w-80">
							<Popover.Title
								>Capture from {selectedHistoryEntry
									? formatHistoryDate(selectedHistoryEntry.capturedAt)
									: "history"}</Popover.Title
							>
							{#if historyImage.isPending}
								<div class="flex aspect-video items-center justify-center rounded-lg bg-muted">
									<Spinner />
								</div>
							{:else if historyImage.isError}
								<p class="text-xs text-destructive">Unable to load this capture.</p>
							{:else if hasHistoryFeed}
								<img
									src={historyUrl}
									alt={`Capture from ${cameraName} at ${formatHistoryDate(entry.capturedAt)}`}
									class="max-h-64 w-full rounded-lg object-contain"
								/>
							{:else}
								<p class="text-xs text-muted-foreground">
									No camera feed available for this capture.
								</p>
							{/if}
						</Popover.Content>
					</Popover.Root>
				{/each}
			</div>
			{#if historyQuery.isFetchingNextPage}
				<div class="flex justify-center py-3"><Spinner class="size-4" /></div>
			{/if}
		{/if}
	</div>
</div>

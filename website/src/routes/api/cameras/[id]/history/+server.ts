import { and, desc, eq, isNotNull, lt, or, sql } from "drizzle-orm";
import { z } from "zod";
import { api } from "$lib/server/api";
import { getDB } from "$lib/server/database";
import { cameras, images, predictions } from "$lib/server/schema";
import { historyPageSchema } from "$lib/types/history";
import { PredictionStatus } from "$lib/types/image-processing";

const DEFAULT_LIMIT = 20;
const MAX_LIMIT = 50;
// Client-local midnight can be on the previous UTC date, so allow one day of timezone offset.
const MINIMUM_HISTORY_DATE = new Date("2026-09-24T00:00:00.000Z");

function maximumHistoryDate(): Date {
	const date = new Date();
	date.setUTCDate(date.getUTCDate() + 1);
	date.setUTCHours(23, 59, 59, 999);
	return date;
}

const historyQuerySchema = z
	.object({
		limit: z.coerce.number().int().min(1).max(MAX_LIMIT).default(DEFAULT_LIMIT),
		cursor: z.string().optional(),
		from: z.iso
			.datetime()
			.optional()
			.transform((value) => (value ? new Date(value) : undefined)),
		to: z.iso
			.datetime()
			.optional()
			.transform((value) => (value ? new Date(value) : undefined)),
	})
	.superRefine((value, context) => {
		const maximumDate = maximumHistoryDate();
		if (value.from && value.from < MINIMUM_HISTORY_DATE) {
			context.addIssue({
				code: "custom",
				path: ["from"],
				message: "History starts on September 25, 2026",
			});
		}
		if (value.from && value.from > maximumDate) {
			context.addIssue({
				code: "custom",
				path: ["from"],
				message: "History cannot be in the future",
			});
		}
		if (value.to && value.to < MINIMUM_HISTORY_DATE) {
			context.addIssue({
				code: "custom",
				path: ["to"],
				message: "History starts on September 25, 2026",
			});
		}
		if (value.to && value.to > maximumDate) {
			context.addIssue({
				code: "custom",
				path: ["to"],
				message: "History cannot be in the future",
			});
		}
		if (value.from && value.to && value.from > value.to) {
			context.addIssue({
				code: "custom",
				path: ["to"],
				message: "The end must be after the start",
			});
		}
	});

type Cursor = { capturedAt: string; id: string };

function encodeCursor(cursor: Cursor): string {
	return Buffer.from(JSON.stringify(cursor)).toString("base64url");
}

function decodeCursor(value: string): Cursor {
	const parsed = JSON.parse(Buffer.from(value, "base64url").toString("utf8")) as Partial<Cursor>;
	if (!parsed.capturedAt || !parsed.id) throw new Error("Invalid history cursor");
	if (!z.uuid().safeParse(parsed.id).success || Number.isNaN(Date.parse(parsed.capturedAt))) {
		throw new Error("Invalid history cursor");
	}
	return { capturedAt: parsed.capturedAt, id: parsed.id };
}

export const GET = api({
	id: "cameras.history",
	name: "Get camera history",
	description: "Get paginated camera image history",
	schema: {
		params: z.object({ id: z.uuid() }),
		query: historyQuerySchema,
		output: historyPageSchema,
	},
	handle: async ({ params, query }) => {
		const db = getDB();
		const camera = await db
			.select({ id: cameras.id })
			.from(cameras)
			.where(eq(cameras.id, params.id))
			.limit(1);
		if (!camera[0]) return { entries: [], nextCursor: null };

		const latestPredictions = db
			.selectDistinctOn([predictions.imageId], {
				imageId: predictions.imageId,
				status: predictions.status,
				alertStatus: predictions.alertStatus,
				confidence: predictions.confidence,
			})
			.from(predictions)
			.orderBy(predictions.imageId, desc(predictions.createdAt))
			.as("latest_predictions");

		const cursor = query.cursor ? decodeCursor(query.cursor) : null;
		const conditions = [eq(images.cameraId, params.id), isNotNull(images.capturedAt)];
		if (query.from) conditions.push(sql`${images.capturedAt} >= ${query.from}`);
		if (query.to) conditions.push(sql`${images.capturedAt} <= ${query.to}`);
		if (cursor) {
			conditions.push(
				or(
					lt(images.capturedAt, new Date(cursor.capturedAt)),
					and(eq(images.capturedAt, new Date(cursor.capturedAt)), lt(images.id, cursor.id)),
				)!,
			);
		}

		const rows = await db
			.select({
				id: images.id,
				capturedAt: images.capturedAt,
				fetchedAt: images.fetchedAt,
				processingStatus: images.processingStatus,
				predictionStatus: latestPredictions.status,
				alertStatus: latestPredictions.alertStatus,
				predictionConfidence: latestPredictions.confidence,
			})
			.from(images)
			.leftJoin(latestPredictions, eq(images.id, latestPredictions.imageId))
			.where(and(...conditions))
			.orderBy(sql`${images.capturedAt} desc`, desc(images.id))
			.limit(query.limit + 1);

		const hasMore = rows.length > query.limit;
		const entries = rows.slice(0, query.limit).map((row) => ({
			...row,
			capturedAt: row.capturedAt!,
			predictionStatus: row.predictionStatus as PredictionStatus | null,
			alertStatus: (row.alertStatus ?? row.predictionStatus) as PredictionStatus | null,
		}));
		const last = entries.at(-1);

		return {
			entries,
			nextCursor:
				hasMore && last
					? encodeCursor({ capturedAt: last.capturedAt.toISOString(), id: last.id })
					: null,
		};
	},
});

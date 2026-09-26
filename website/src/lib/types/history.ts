import z from "zod";
import { PredictionStatus } from "./image-processing";

export const historyEntrySchema = z.object({
	id: z.uuid(),
	capturedAt: z.date(),
	fetchedAt: z.date(),
	processingStatus: z.string(),
	predictionStatus: z.enum(PredictionStatus).nullable(),
	alertStatus: z.enum(PredictionStatus).nullable(),
	predictionConfidence: z.string().nullable(),
});

export const historyPageSchema = z.object({
	entries: z.array(historyEntrySchema),
	nextCursor: z.string().nullable(),
});

export type HistoryEntry = z.infer<typeof historyEntrySchema>;
export type HistoryPage = z.infer<typeof historyPageSchema>;

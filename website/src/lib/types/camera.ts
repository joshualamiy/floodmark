import z from "zod";
import { PredictionStatus } from "./image-processing";

export const cameraSchema = z.object({
	id: z.uuid(),
	source: z.string(),
	sourceCameraId: z.string(),
	sourceId: z.string().nullable(),
	name: z.string(),
	roadway: z.string().nullable(),
	direction: z.string().nullable(),
	locationDescription: z.string().nullable(),
	latitude: z.string(),
	longitude: z.string(),
	sortOrder: z.number().int().nullable(),
	sourceViewId: z.string().nullable(),
	sourceUrl: z.string().nullable(),
	sourceViewStatus: z.string().nullable(),
	sourceViewDescription: z.string().nullable(),
	sourceViewSortId: z.number().int().nullable(),
	isActive: z.boolean(),
	createdAt: z.date(),
	updatedAt: z.date(),
	latestImage: z
		.object({
			id: z.uuid(),
			s3Key: z.string(),
			capturedAt: z.date().nullable(),
			processingStatus: z.string(),
			processedAt: z.date().nullable(),
			processingError: z.string().nullable(),
			predictionStatus: z.enum(PredictionStatus).nullable(),
			alertStatus: z.enum(PredictionStatus).nullable(),
			alertNote: z.string().nullable(),
			predictionConfidence: z.string().nullable(),
			predictionNote: z.string().nullable(),
			heatmapS3Key: z.string().nullable(),
			heatmapStatus: z.string().nullable(),
			heatmapNote: z.string().nullable(),
		})
		.nullable(),
});

export type Camera = z.infer<typeof cameraSchema>;

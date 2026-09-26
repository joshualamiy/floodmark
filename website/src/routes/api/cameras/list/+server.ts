import { asc, desc, eq, sql } from "drizzle-orm";
import { z } from "zod";
import { api } from "$lib/server/api";
import { getDB } from "$lib/server/database";
import { cameras, images, predictions } from "$lib/server/schema";
import { cameraSchema } from "$lib/types/camera";
import { PredictionStatus } from "$lib/types/image-processing";

export const GET = api({
	id: "cameras.list",
	name: "List cameras",
	description: "List active cameras ordered by their configured sort order",
	schema: {
		output: z.array(cameraSchema),
	},
	handle: async () => {
		const db = getDB();
		const latestImages = db
			.selectDistinctOn([images.cameraId], {
				cameraId: images.cameraId,
				id: images.id,
				s3Key: images.s3Key,
				capturedAt: images.capturedAt,
				processingStatus: images.processingStatus,
				processedAt: images.processedAt,
				processingError: images.processingError,
			})
			.from(images)
			.orderBy(images.cameraId, sql`${images.capturedAt} desc nulls last`, desc(images.fetchedAt))
			.as("latest_images");
		const latestPredictions = db
			.selectDistinctOn([predictions.imageId], {
				imageId: predictions.imageId,
				status: predictions.status,
				confidence: predictions.confidence,
				heatmapS3Key: predictions.heatmapS3Key,
			})
			.from(predictions)
			.orderBy(predictions.imageId, desc(predictions.createdAt))
			.as("latest_predictions");

		const rows = await db
			.select({
				camera: cameras,
				latestImage: {
					id: latestImages.id,
					s3Key: latestImages.s3Key,
					capturedAt: latestImages.capturedAt,
					processingStatus: latestImages.processingStatus,
					processedAt: latestImages.processedAt,
					processingError: latestImages.processingError,
					predictionStatus: latestPredictions.status,
					predictionConfidence: latestPredictions.confidence,
					heatmapS3Key: latestPredictions.heatmapS3Key,
				},
			})
			.from(cameras)
			.leftJoin(latestImages, eq(cameras.id, latestImages.cameraId))
			.leftJoin(latestPredictions, eq(latestImages.id, latestPredictions.imageId))
			.where(eq(cameras.isActive, true))
			.orderBy(asc(cameras.sortOrder), asc(cameras.name));

		return rows.map((result) => ({
			...result.camera,
			latestImage: result.latestImage.id
				? {
						id: result.latestImage.id,
						s3Key: result.latestImage.s3Key!,
						capturedAt: result.latestImage.capturedAt,
						processingStatus: result.latestImage.processingStatus!,
						processedAt: result.latestImage.processedAt,
						processingError: result.latestImage.processingError,
						predictionStatus: result.latestImage.predictionStatus as PredictionStatus | null,
						predictionConfidence: result.latestImage.predictionConfidence,
						heatmapS3Key: result.latestImage.heatmapS3Key,
					}
				: null,
		}));
	},
});

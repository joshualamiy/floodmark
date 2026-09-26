import { GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { error } from "@sveltejs/kit";
import { desc, eq } from "drizzle-orm";
import { z } from "zod";
import { api } from "$lib/server/api";
import { getDB } from "$lib/server/database";
import { images, predictions } from "$lib/server/schema";
import { getStorage } from "$lib/server/storage";

const URL_EXPIRY_SECONDS = 300;

export const GET = api({
	id: "cameras.snapshot",
	name: "Get camera snapshot URL",
	description: "Get a presigned URL for the latest camera snapshot or heatmap",
	schema: {
		params: z.object({ id: z.uuid() }),
		query: z.object({
			heatmap: z
				.enum(["true", "false"])
				.default("false")
				.transform((value) => value === "true"),
		}),
		output: z.object({
			url: z.url(),
			expiresAt: z.iso.datetime(),
		}),
	},
	handle: async ({ params, query }) => {
		const db = getDB();
		const [image] = await db
			.select({
				id: images.id,
				bucket: images.s3Bucket,
				key: images.s3Key,
				capturedAt: images.capturedAt,
			})
			.from(images)
			.where(eq(images.cameraId, params.id))
			.orderBy(desc(images.capturedAt), desc(images.fetchedAt))
			.limit(1);

		if (!image) error(404, "Camera snapshot not found");

		let key = image.key;
		if (query.heatmap) {
			const [prediction] = await db
				.select({ key: predictions.heatmapS3Key })
				.from(predictions)
				.where(eq(predictions.imageId, image.id))
				.orderBy(desc(predictions.createdAt))
				.limit(1);

			if (!prediction?.key) error(404, "Camera heatmap not found");
			key = prediction.key;
		}

		const expiresAt = new Date(Date.now() + URL_EXPIRY_SECONDS * 1000);
		const { storage } = getStorage();
		const url = await getSignedUrl(
			storage,
			new GetObjectCommand({ Bucket: image.bucket, Key: key }),
			{ expiresIn: URL_EXPIRY_SECONDS },
		);

		return {
			url,
			expiresAt: expiresAt.toISOString(),
		};
	},
});

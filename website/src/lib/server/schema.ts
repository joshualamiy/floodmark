import {
	boolean,
	index,
	integer,
	jsonb,
	numeric,
	pgEnum,
	pgTable,
	text,
	timestamp,
	uniqueIndex,
	uuid,
} from "drizzle-orm/pg-core";
import { imageProcessingStatuses } from "$lib/types/image-processing";

export const imageProcessingStatus = pgEnum("image_processing_status", imageProcessingStatuses);

export const predictionStatus = pgEnum("prediction_status", ["dry", "wet", "flooded"]);

export const cameras = pgTable(
	"cameras",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		source: text("source").notNull(),
		sourceCameraId: text("source_camera_id").notNull(),
		sourceId: text("source_id"),

		name: text("name").notNull(),
		roadway: text("roadway"),
		direction: text("direction"),
		locationDescription: text("location_description"),
		latitude: numeric("latitude", { precision: 9, scale: 6 }).notNull(),
		longitude: numeric("longitude", { precision: 9, scale: 6 }).notNull(),
		sortOrder: integer("sort_order"),

		sourceViewId: text("source_view_id"),
		sourceUrl: text("source_url"),
		sourceViewStatus: text("source_view_status"),
		sourceViewDescription: text("source_view_description"),
		sourceViewSortId: integer("source_view_sort_id"),

		isActive: boolean("is_active").notNull().default(true),
		createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
		updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
	},
	(table) => [
		uniqueIndex("cameras_source_camera_unique").on(table.source, table.sourceCameraId),
		index("cameras_active_idx").on(table.isActive),
	],
);

export const images = pgTable(
	"images",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		cameraId: uuid("camera_id")
			.notNull()
			.references(() => cameras.id, { onDelete: "cascade" }),

		sourceUrl: text("source_url"),
		r2Bucket: text("r2_bucket").notNull(),
		r2Key: text("r2_key").notNull(),
		contentType: text("content_type"),
		byteSize: integer("byte_size"),
		sha256: text("sha256"),

		capturedAt: timestamp("captured_at", { withTimezone: true }),
		fetchedAt: timestamp("fetched_at", { withTimezone: true }).notNull(),

		processingStatus: imageProcessingStatus("processing_status").notNull().default("unprocessed"),
		processedAt: timestamp("processed_at", { withTimezone: true }),
		processingError: text("processing_error"),

		createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
	},
	(table) => [
		uniqueIndex("images_r2_object_unique").on(table.r2Bucket, table.r2Key),
		index("images_camera_captured_idx").on(table.cameraId, table.capturedAt),
		index("images_processing_queue_idx").on(table.processingStatus, table.createdAt),
		index("images_sha256_idx").on(table.sha256),
	],
);

export const predictions = pgTable(
	"predictions",
	{
		id: uuid("id").defaultRandom().primaryKey(),
		imageId: uuid("image_id")
			.notNull()
			.references(() => images.id, { onDelete: "cascade" }),

		modelName: text("model_name").notNull(),
		modelVersion: text("model_version").notNull(),
		status: predictionStatus("status").notNull(),
		confidence: numeric("confidence", { precision: 5, scale: 4 }).notNull(),
		stageProbabilities: jsonb("stage_probabilities")
			.$type<Record<"dry" | "wet" | "flooded", number>>()
			.notNull(),
		heatmapR2Key: text("heatmap_r2_key"),

		inferenceStartedAt: timestamp("inference_started_at", { withTimezone: true }),
		inferenceCompletedAt: timestamp("inference_completed_at", { withTimezone: true }),
		createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
	},
	(table) => [
		uniqueIndex("predictions_image_model_unique").on(
			table.imageId,
			table.modelName,
			table.modelVersion,
		),
		index("predictions_status_idx").on(table.status),
	],
);

CREATE TYPE "image_processing_status" AS ENUM('unprocessed', 'processed', 'error', 'skipped');--> statement-breakpoint
CREATE TYPE "prediction_status" AS ENUM('dry', 'wet', 'flooded');--> statement-breakpoint
CREATE TABLE "camera_notification_emails" (
	"camera_id" uuid,
	"email_id" uuid,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	CONSTRAINT "camera_notification_emails_pk" PRIMARY KEY("camera_id","email_id")
);
--> statement-breakpoint
CREATE TABLE "cameras" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid(),
	"source" text NOT NULL,
	"source_camera_id" text NOT NULL,
	"source_id" text,
	"name" text NOT NULL,
	"roadway" text,
	"direction" text,
	"location_description" text,
	"latitude" numeric(9,6) NOT NULL,
	"longitude" numeric(9,6) NOT NULL,
	"sort_order" integer,
	"source_view_id" text,
	"source_url" text,
	"source_view_status" text,
	"source_view_description" text,
	"source_view_sort_id" integer,
	"is_active" boolean DEFAULT true NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "images" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid(),
	"camera_id" uuid NOT NULL,
	"source_url" text,
	"s3_bucket" text NOT NULL,
	"s3_key" text NOT NULL,
	"content_type" text,
	"byte_size" integer,
	"sha256" text,
	"captured_at" timestamp with time zone,
	"fetched_at" timestamp with time zone NOT NULL,
	"processing_status" "image_processing_status" DEFAULT 'unprocessed'::"image_processing_status" NOT NULL,
	"processed_at" timestamp with time zone,
	"processing_error" text,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "notification_emails" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid(),
	"email" text NOT NULL,
	"verified_at" timestamp with time zone,
	"confirmed" boolean DEFAULT false NOT NULL,
	"verification_token_hash" text,
	"verification_token_expires_at" timestamp with time zone,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "predictions" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid(),
	"image_id" uuid NOT NULL,
	"model_version" jsonb NOT NULL,
	"status" "prediction_status" NOT NULL,
	"confidence" numeric(5,4) NOT NULL,
	"stage_a_probabilities" jsonb NOT NULL,
	"stage_b_probabilities" jsonb NOT NULL,
	"stage_probabilities" jsonb NOT NULL,
	"thresholds" jsonb NOT NULL,
	"note" text,
	"heatmap_r2_key" text,
	"inference_started_at" timestamp with time zone,
	"inference_completed_at" timestamp with time zone,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE INDEX "camera_notification_emails_email_idx" ON "camera_notification_emails" ("email_id");--> statement-breakpoint
CREATE UNIQUE INDEX "cameras_source_camera_unique" ON "cameras" ("source","source_camera_id");--> statement-breakpoint
CREATE INDEX "cameras_active_idx" ON "cameras" ("is_active");--> statement-breakpoint
CREATE UNIQUE INDEX "images_r2_object_unique" ON "images" ("s3_bucket","s3_key");--> statement-breakpoint
CREATE INDEX "images_camera_captured_idx" ON "images" ("camera_id","captured_at");--> statement-breakpoint
CREATE INDEX "images_processing_queue_idx" ON "images" ("processing_status","created_at");--> statement-breakpoint
CREATE INDEX "images_sha256_idx" ON "images" ("sha256");--> statement-breakpoint
CREATE UNIQUE INDEX "notification_emails_email_unique" ON "notification_emails" ("email");--> statement-breakpoint
CREATE UNIQUE INDEX "predictions_image_model_unique" ON "predictions" ("image_id","model_version");--> statement-breakpoint
CREATE INDEX "predictions_status_idx" ON "predictions" ("status");--> statement-breakpoint
ALTER TABLE "camera_notification_emails" ADD CONSTRAINT "camera_notification_emails_camera_id_cameras_id_fkey" FOREIGN KEY ("camera_id") REFERENCES "cameras"("id") ON DELETE CASCADE;--> statement-breakpoint
ALTER TABLE "camera_notification_emails" ADD CONSTRAINT "camera_notification_emails_email_id_notification_emails_id_fkey" FOREIGN KEY ("email_id") REFERENCES "notification_emails"("id") ON DELETE CASCADE;--> statement-breakpoint
ALTER TABLE "images" ADD CONSTRAINT "images_camera_id_cameras_id_fkey" FOREIGN KEY ("camera_id") REFERENCES "cameras"("id") ON DELETE CASCADE;--> statement-breakpoint
ALTER TABLE "predictions" ADD CONSTRAINT "predictions_image_id_images_id_fkey" FOREIGN KEY ("image_id") REFERENCES "images"("id") ON DELETE CASCADE;
import { env } from "$env/dynamic/private";
import { S3Client } from "@aws-sdk/client-s3";

export type Storage = {
	bucket: string;
	storage: S3Client;
};

let client: Storage | undefined;

export function getStorage(): Storage {
	if (!client) {
		client = {
			bucket: env.S3_BUCKET ?? "imagesss",
			storage: new S3Client({
				endpoint: env.S3_ENDPOINT_URL,
				region: env.S3_REGION,
				credentials: {
					accessKeyId: env.S3_ACCESS_KEY_ID,
					secretAccessKey: env.S3_SECRET_ACCESS_KEY,
				},
			}),
		};
	}

	return client;
}

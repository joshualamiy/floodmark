export const imageProcessingStatuses = [
	"unprocessed",
	"processed",
	"error",
	"skipped",
] as const;

export type ImageProcessingStatus = (typeof imageProcessingStatuses)[number];

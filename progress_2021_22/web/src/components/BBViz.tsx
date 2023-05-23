import type { FC } from 'react';

import { useState, useEffect, useRef } from 'react';

interface IBBVizProps {
  image: string;
  boxes: number[][];
  segmentationJsonUrl?: string;
  onSelected?: (index: number) => void;
  options: {
    colors?: {
      normal?: string;
      selected?: string;
      unselected?: string;
    };
    style?: Record<string, any>;
    styleSegmentation?: Record<string, any>;
    base64Image?: boolean;
  };
  imageSize?: [number, number];
  onClick?: (e: MouseEvent) => void;
}

const BBViz: FC<Partial<IBBVizProps>> = ({
  image,
  boxes,
  segmentationJsonUrl,
  onSelected,
  onClick,
  options,
  imageSize,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [isSegmented, setIsSegmented] = useState<boolean>(false);
  const [separateSegmentationProp, setSeparateSegmentationProp] =
    useState<boolean>(false);
  const [pixelSegmentation, setPixelSegmentation] = useState<number[]>([]);

  const renderBox = (box: number[], index: number, color: string) => {
    if (!box || typeof box === 'undefined') return null;

    let lineWidth = 2;
    if (canvasRef?.current && canvasRef.current?.width > 600) lineWidth = 3;
    if (canvasRef?.current && canvasRef?.current?.width > 1000) lineWidth = 4;

    drawBox(box, color, lineWidth);
  };

  const drawBox = (box: number[], color: string, lineWidth: number) => {
    if (!box || typeof box === 'undefined') return null;

    if (!canvasRef?.current) return null;

    const canvas = canvasRef.current;

    const ctx = canvasRef.current?.getContext('2d');
    if (!ctx) return null;

    let coord = box;

    let [x, y, width, height, label] = coord;

    if (x < lineWidth / 2) {
      x = lineWidth / 2;
    }
    if (y < lineWidth / 2) {
      y = lineWidth / 2;
    }

    if (x + width > canvas.width) {
      width = canvas.width - lineWidth - x;
    }
    if (y + height > canvas.height) {
      height = canvas.height - lineWidth - y;
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(x + width, y);
    ctx.lineTo(x, y);
    ctx.lineTo(x, y + height);
    ctx.lineTo(x + width, y + height);
    ctx.lineTo(x + width, y);
    ctx.stroke();

    ctx.font = '18px Arial';
    ctx.fillStyle = 'rgba(255, 0, 255, 1)';

    const text = label.toString();

    ctx.fillText(text, x + width, y);
  };

  const renderBoxes = () => {
    if (typeof boxes === 'undefined') return;
    if (boxes === null) boxes = [];

    if (boxes.length && !Array.isArray(boxes[0])) return;

    const uniqueClasses = Array.from(new Set(boxes.map((box) => box[4])));
    const totalClasses = uniqueClasses.length;

    // Generate colors for each class
    const colors: string[] = [];
    for (let i = 0; i < totalClasses; i++) {
      colors.push(`#${Math.floor(Math.random() * 16777215).toString(16)}`);
    }

    boxes
      .map((box, index) => {
        const selected = index === hoverIndex;
        return { box, index, selected };
      })
      .sort((a) => (a.selected ? 1 : -1))
      .forEach((box) =>
        renderBox(box.box, box.index, colors[uniqueClasses.indexOf(box.box[4])])
      );
  };

  useEffect(() => {
    if (!imageSize) imageSize = [640, 480];

    if (segmentationJsonUrl) {
      fetch(segmentationJsonUrl)
        .then((res) => res.json())
        .then((json) => {
          if (
            json.body &&
            json.body.predictions &&
            json.body.predictions.length &&
            json.body.predictions[0]?.vals.length
          ) {
            setIsSegmented(false);
          }
        });
    }

    if (canvasRef.current !== null && imageSize) {
      const ctx = canvasRef.current.getContext('2d');
      if (!ctx) return;

      const bg = new Image();
      if (!image) throw new Error('No image');
      bg.src = options?.base64Image ? 'data:image/png;base64,' + image : image;

      bg.onload = () => {
        if (!canvasRef.current) return;
        if (!imageSize) return;

        if (bg.height > imageSize[1]) bg.height = imageSize[1];
        if (bg.width > imageSize[0]) bg.width = imageSize[0];

        canvasRef.current.width = bg.width;
        canvasRef.current.height = bg.height;

        ctx.drawImage(bg, 0, 0, imageSize[0], imageSize[1]);
        renderBoxes();

        canvasRef.current.onmousemove = (e) => {
          if (!canvasRef.current) return;
          const r = canvasRef.current?.getBoundingClientRect();
          const scaleX = canvasRef.current.width / r.width;
          const scaleY = canvasRef.current.height / r.height;
          const x = (e.clientX - r.left) * scaleX;
          const y = (e.clientY - r.top) * scaleY;

          const selectedBox: {
            index: number;
            dimensions: any;
          } = { index: -1, dimensions: null };

          if (boxes && boxes.length) {
            boxes.forEach((box, index) => {
              if (!box || typeof box === 'undefined') return null;

              const coord = box;

              let [bx, by, bw, bh] = [0, 0, 0, 0];

              // coord is an array containing x, y, width, height
              [bx, by, bw, bh] = coord;

              if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) {
                // The mouse honestly hits the rect
                const insideBoxFlag =
                  !selectedBox.dimensions ||
                  (bx >= selectedBox.dimensions[0] &&
                    bx <=
                      selectedBox.dimensions[0] + selectedBox.dimensions[2] &&
                    by >= selectedBox.dimensions[1] &&
                    by <=
                      selectedBox.dimensions[1] + selectedBox.dimensions[3]);
                if (insideBoxFlag) {
                  selectedBox.index = index;
                  selectedBox.dimensions = box;
                }
              }
            });
          } else if (pixelSegmentation && pixelSegmentation.length) {
            selectedBox.index = 0;
            selectedBox.dimensions =
              pixelSegmentation[x + canvasRef.current.width * y];
          }

          if (onSelected) onSelected(selectedBox.index);
          setHoverIndex(selectedBox.index);
        };

        canvasRef.current.onmouseout = () => {
          if (onSelected) onSelected(-1);
          setHoverIndex(-1);
        };

        canvasRef.current.onclick = (e) => {
          if (onClick) onClick(e);
        };
      };
    }
  }, []);

  return (
    <div
      className="bbox-viz"
      style={{
        display: 'flex',
        justifyContent: 'center',
      }}
    >
      <canvas className="bbox-canvas" ref={canvasRef} />
    </div>
  );
};

export default BBViz;

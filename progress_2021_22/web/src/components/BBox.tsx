import { useEffect, useRef, useState } from 'react';
import { toast } from 'react-toastify';

interface IBox {
  centerX: number;
  centerY: number;
  width: number;
  height: number;
  type: number;
}

export default ({
  imgSrc,
  imgSize,
  types,
  fName,
}: {
  imgSrc: string;
  imgSize: {
    height: number;
    width: number;
  };
  types: string[] | number[];
  fName: string;
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [boxes, setBoxes] = useState<IBox[]>(
    JSON.parse(localStorage.getItem(`${imgSrc.split('/').pop()}`) || '[]') || []
  );
  const [currentBox, setCurrentBox] = useState<number>(-1);
  const [currentSelection, setCurrentSelection] = useState<number>(-1);
  const [click, setClick] = useState<{ x: number; y: number } | undefined>(
    undefined
  );

  const [uniqueClasses, setUniqueClasses] = useState<number[]>([]);
  const [colors, setColors] = useState<string[]>([]);

  useEffect(() => {
    const localTypes = types.map((s) =>
      typeof s === 'number' ? s : parseInt(s, 10)
    );

    const lUniqueClasses = Array.from(new Set(localTypes));
    const totalClasses = lUniqueClasses.length;

    setUniqueClasses(lUniqueClasses);

    // Generate colors for each class
    const localColors: string[] = [];
    for (let i = 0; i < totalClasses; i++) {
      localColors.push(`#${Math.floor(Math.random() * 16777215).toString(16)}`);
    }

    setColors(localColors);

    if (!canvasRef?.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) return;
    const img = new Image();
    img.src = imgSrc;

    img.onload = () => {
      canvas.width = imgSize.width;
      canvas.height = imgSize.height;
      ctx.drawImage(img, 0, 0, imgSize.width, imgSize.height);
    };
  }, [imgSrc]);

  const clearBoxes = () => {
    return new Promise((resolve) => {
      if (!canvasRef?.current) return;

      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      if (!ctx) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Re-draw the image:
      const img = new Image();
      img.src = imgSrc;

      img.onload = () => {
        canvas.width = imgSize.width;
        canvas.height = imgSize.height;
        ctx.drawImage(img, 0, 0, imgSize.width, imgSize.height);
        resolve(0);
      };
    });
  };

  const renderBox = (box: IBox, color: string) => {
    if (!canvasRef?.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    let lineWidth = 2;
    if (canvas.width > 600) lineWidth = 3;
    if (canvas.width > 1000) lineWidth = 4;
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();

    const { centerX, centerY, width, height } = box;

    const x = centerX - width / 2;
    const y = centerY - height / 2;
    ctx.moveTo(x + width, y);
    ctx.lineTo(x, y);
    ctx.lineTo(x, y + height);
    ctx.lineTo(x + width, y + height);
    ctx.lineTo(x + width, y);
    ctx.stroke();

    ctx.closePath();

    ctx.font = '14px Arial';
    ctx.fillStyle = color;
    const text = `${box.type}`;
    const textWidth = ctx.measureText(text).width;
    const textHeight = 14;
    ctx.fillText(text, centerX - textWidth / 2, centerY + textHeight / 2);
  };

  const renderBoxes = (boxes: IBox[]) => {
    if (!boxes.length) return;

    boxes.forEach((box, i) => {
      renderBox(box, colors[uniqueClasses.indexOf(box.type)]);
    });
  };

  return (
    <>
      <div
        style={{
          display: 'flex',
          flexDirection: 'row',
          minWidth: '100%',
        }}
      >
        <div style={{ padding: '20px' }}>
          <h3>Select Type to Draw</h3>
          <select
            style={{
              minWidth: '100px',
            }}
            onChange={(e) => {
              const type = e.target.value;
              setCurrentSelection(parseInt(type, 10));
            }}
          >
            <option value="-1">Select Type</option>
            {types.map((type, index) => (
              <option key={index} value={index}>
                {type}
              </option>
            ))}
          </select>

          <button
            onClick={() => {
              // Remove the last box in the array
              if (boxes.length) {
                let tempBoxes = boxes;
                tempBoxes.pop();
                setBoxes(tempBoxes);

                // Redraw the boxes:
                clearBoxes().then(() => {
                  renderBoxes([...tempBoxes]);
                });
              }
            }}
          >
            Undo Last Box
          </button>
        </div>
        <canvas
          ref={canvasRef}
          style={{
            aspectRatio: '4 / 3',
          }}
          onClick={(e) => {
            if (currentSelection === -1) {
              toast('Please select a type first', { type: 'error' });
              return;
            }

            if (!canvasRef.current) return;
            const r = canvasRef.current?.getBoundingClientRect();
            const scaleX = canvasRef.current.width / r.width;
            const scaleY = canvasRef.current.height / r.height;
            let x = (e.clientX - r.left) * scaleX;
            let y = (e.clientY - r.top) * scaleY;

            // Floor the x and y values to get the pixel coordinates of the mouse click.
            x = Math.floor(x);
            y = Math.floor(y);

            if (click) {
              let width = Math.abs(x - click.x);
              let height = Math.abs(y - click.y);

              const topLeftX = Math.min(click.x, x);
              const topLeftY = Math.min(click.y, y);

              // Calculate the center of the box.
              const centerX = topLeftX + width / 2;
              const centerY = topLeftY + height / 2;

              const newBox: IBox = {
                centerX,
                centerY,
                width,
                height,
                type: currentSelection + 1,
              };

              setBoxes((pBoxes) => [...pBoxes, newBox]);
              setClick(undefined);

              clearBoxes().then(() => {
                renderBoxes([...boxes, newBox]);
              });
            } else {
              setClick({ x, y });
            }
          }}
        />
      </div>
      <div
        className="save"
        style={{
          minWidth: '100%',
          display: 'flex',
          flexDirection: 'row',
          justifyContent: 'center',
          padding: '20px 0',
        }}
      >
        <button
          style={{
            minWidth: '100px',
            minHeight: '50px',
          }}
          onClick={() => {
            // We want to save the boxes as a yolov3 format:
            // class id, center x, center y, width, height
            // The center x and y are the center of the box normalized to the image size.

            const lines = boxes.map((box) => {
              const normalizedX = box.centerX / imgSize.width;
              const normalizedY = box.centerY / imgSize.height;

              return `${box.type - 1} ${normalizedX} ${normalizedY} ${
                box.width
              } ${box.height}`;
            });

            const blob = new Blob([lines.join('\n')], { type: 'text/plain' });

            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${fName}.txt`;
            a.click();
            URL.revokeObjectURL(url);
          }}
        >
          Save
        </button>
      </div>
    </>
  );
};

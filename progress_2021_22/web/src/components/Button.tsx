import type { FC, MouseEventHandler } from "react"

interface IButtonProps {
  children?: React.ReactNode;
  onClick?: MouseEventHandler<HTMLDivElement> | undefined;
  className?: string;
  style?: Record<string, any>
}

const Button: FC<IButtonProps> = ({ children, onClick, className, ...props }) => (
  <div onClick={onClick} className={className} style={{
    cursor: onClick ? "pointer" : "default",
    display: "inline-block",
    padding: "0.5em 1em",
    backgroundColor: "white",
    border: "1px solid black",
    borderRadius: "0.25em",
    color: "black",
    margin: "0.5em",
    ...props.style
  }}>
    {children}
  </div>
)

export default Button